import re
from time import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl
import wget
import yaml
from tqdm import tqdm
from loguru import logger

from neuromechants import get_bulkdata_dir

# URLs and regex patterns for parsing the BIOMEDISA AntScan pages
BIOMEDISA_INDEX_PAGE_URL = "https://biomedisa.info/antscan/?show_all=True"
BIOMEDISA_details_page_URL_TEMPLATE = "https://biomedisa.info/antscan/specimen/{id_}/"
BIOMEDISA_DOWNLOAD_URL_TEMPLATE = (
    "https://biomedisa.info/antscan/download/?id={id_}&object=processed"
)

# For parsing patterns like this:
# <td>
# <div id="img560" >
#     <p style="text-align:left;">
#     <!-- screenshot -->
#     <a href="/antscan/specimen/560"><img src="/media/antscan/processed/5x/24-43.png" title="info" style="width:425px;height:238px;border:0"></a>
#     </br></br>
#     <!-- get name of specimen -->
#         <span>Acanthognathus cf.ocellatus | Myrmicinae | male | CASENT0744649</span>
#     <!-- feature buttons --></br>
#     <a href=# onclick="downloadFunction(560,'specimen','antscan')"><img src="/static/downloads.png" alt="download" style="width:20px;height:20px;border:0;"></a>
#     <a href=# onclick="sliceviewerFunction(560,'antscan')"><img src="/static/2D_plain.png" title="slice viewer" width="30" height="20" onMouseOver="this.src='/static/2D_filled.png'" onMouseOut="this.src='/static/2D_plain.png'"></a>
#     <a href=# onclick="visualizationFunction(560,'antscan')"><img src="/static/mesh.svg" title="visualization" style="width:20px;height:20px;"></a>
#     <a href="/antscan/specimen/560"><img src="/static/info.png" alt="info" style="width:20px;height:20px;border:0;"></a>
#     </p>
# </div>
# </td>
RE_PATTERN_INDEX_PAGE = re.compile(
    r"<td>"
    r"[\s\S]+?"
    r"<div id=\"img(?P<imgid_from_div>\d+?)\"\s*?>"
    r"[\s\S]+?"
    r"<span>(?P<name>.+?) \| (?P<subfamily>.+?) \| (?P<caste>.+?) \| (?P<specimen_code>.+?)</span>"
    r"[\s\S]+?"
    r"<a href=\"/antscan/specimen/(?P<id_from_infolink>\d+?)\">"
    r"<img src=\"/static/info\.png\" alt=\"info\" style=\".+?\"></a>"
    r"[\s\S]+?"
    r"</td>"
)

# For parsing patterns like this:
# <h2> <span> Processed data </span> </h2>
#     <div id="img2191" >
#         <p style="text-align:left;">
#         <a href=# onclick="visualizationFunction(2191,'processed','antscan')"><img src="/static/file_mesh.svg" title="visualization" style="width:50px;height:50px;border:0;"></a>
#         <a href=# onclick="downloadFunction(2191,'processed','antscan')"><img src="/static/downloads.png" title="download" style="width:20px;height:20px;border:0;"></a>
#         <a href=# onclick="visualizationFunction(2191,'processed','antscan')"><img src="/static/mesh.svg" title="visualization" style="width:20px;height:20px;"></a>
#         <a href=# onclick="alertFunction('Login in to copy to storage')"><img src="/static/share2.png" title="share" style="width:20px;height:20px;border:0;"></a>
#         antscan/processed/5x/24-43.stl
#         <span style="float:right;"> </br> Jan. 30, 2023, 2:58 p.m.</span>
#         </p>
#     </div>
#     <div id="img12205" >
#         <p style="text-align:left;">
#         <img src="/static/file_image.svg" title="visualization" style="width:50px;height:50px;border:0;">
#         <a href=# onclick="downloadFunction(12205,'processed','antscan')"><img src="/static/downloads.png" alt="download" style="width:20px;height:20px;border:0;"></a>
#         <a href=# onclick="sliceviewerFunction(12205,'processed','antscan')"><img src="/static/2D_plain.png" title="slice viewer" width="30" height="20" onMouseOver="this.src='/static/2D_filled.png'" onMouseOut="this.src='/static/2D_plain.png'"></a>
#         <a href=# onclick="alertFunction('Login in to copy to storage')"><img src="/static/share2.png" title="share" style="width:20px;height:20px;border:0;"></a>
#         antscan/processed/5x/24-43.tif
#         <span style="float:right;"> </br> Feb. 6, 2023, 1:27 p.m.</span>
#         </p>
#     </div>
RE_PATTERN_details_page_MESHFILE = re.compile(
    r"<div id=\"img(?P<imgid_from_div>\d+)\" >"
    r"(?:(?!</div>)[\s\S])*"
    r"/static/file_mesh\.svg"
    r"(?:(?!</div>)[\s\S])*"
    r"downloadFunction\((?P<download_id>\d+),'processed','antscan'\)"
    r"(?:(?!</div>)[\s\S])*"
    r"</div>"
)
RE_PATTERN_details_page_IMAGEFILE = re.compile(
    r"<div id=\"img(?P<imgid_from_div>\d+)\" >"
    r"(?:(?!</div>)[\s\S])*"
    r"/static/file_image\.svg"
    r"(?:(?!</div>)[\s\S])*"
    r"downloadFunction\((?P<download_id>\d+),'processed','antscan'\)"
    r"(?:(?!</div>)[\s\S])*"
    r"</div>"
)


def fetch_index_page_and_get_details_pages(
    index_page_url: str, index_page_path: Path, rewrite_existing: bool = False
) -> pl.DataFrame:
    """Parse links to specimen info pages"""
    if not index_page_path.exists() or rewrite_existing:
        logger.info(f"Downloading the index page from {index_page_url}")
        index_page_path.unlink(missing_ok=True)
        out_path = wget.download(index_page_url, index_page_path.as_posix(), bar=None)
        assert Path(out_path) == index_page_path.absolute()
        logger.success(f"Downloaded the index page to {out_path}")
    else:
        logger.info(f"Index page already exists at {index_page_path}, skip download")

    with open(index_page_path, "r") as f:
        index_page = f.read()
        matches = []
        for match in re.finditer(RE_PATTERN_INDEX_PAGE, index_page):
            assert match.group("imgid_from_div") == match.group("id_from_infolink")
            name = match.group("name")
            subfamily = match.group("subfamily")
            caste = match.group("caste")
            specimen_code = match.group("specimen_code")
            details_page_url = BIOMEDISA_details_page_URL_TEMPLATE.format(
                id_=match.group("id_from_infolink")
            )
            matches.append(
                {
                    "name": name,
                    "subfamily": subfamily if subfamily != "None" else None,
                    "caste": caste if caste != "None" else None,
                    "specimen_code": specimen_code,
                    "details_page_url": details_page_url,
                }
            )
    return pl.DataFrame(matches)


def raise_on_inconsistent_metadata(
    parsed_index_page_df: pl.DataFrame, metadata_df: pl.DataFrame
) -> None:
    """Cross-check the parsed index page with the metadata from the info pages"""
    metadata_df = metadata_df.filter(
        pl.col("specimen_code").is_in(parsed_index_page_df["specimen_code"].implode())
    ).select("specimen_code", "Name", "Subfamily", "caste")
    joint_metadata_df = parsed_index_page_df.join(
        metadata_df, on="specimen_code", how="left"
    )
    inconsistent_rows = joint_metadata_df.filter(
        (pl.col("name") != pl.col("Name"))
        | (pl.col("subfamily") != pl.col("Subfamily"))
        | (pl.col("caste") != pl.col("caste_right"))
    )
    if inconsistent_rows.height > 0:
        logger.error(
            f"Found {inconsistent_rows.height} inconsistent rows between the index "
            f"page and the metadata spreadsheet."
        )
        for row in inconsistent_rows.iter_rows(named=True):
            logger.error(
                "inconsistent row: "
                f"id_on_biomedisa_index={row['id_on_biomedisa_index']}, "
                f"name={row['name']} vs {row['Name']}, "
                f"subfamily={row['subfamily']} vs {row['Subfamily']}, "
                f"caste={row['caste']} vs {row['caste_right']}"
            )
        raise ValueError("Mismatched rows between index page and metadata spreadsheet.")
    else:
        logger.success("Index page and the metadata spreadsheet are all consistent.")


def fetch_details_page_and_get_download_links(details_page_url: str) -> tuple[str, str]:
    with NamedTemporaryFile(suffix=".html") as tempfile:
        Path(tempfile.name).unlink(missing_ok=True)
        wget.download(details_page_url, out=tempfile.name, bar=None)
        with open(tempfile.name, "r") as f:
            html = f.read()

    def _find_single_download_id(re_pattern: re.Pattern, html: str) -> str:
        _mesh_donwload_ids = []
        for match in re.finditer(re_pattern, html):
            download_id = match.group("download_id")
            assert match.group("imgid_from_div") == download_id
            _mesh_donwload_ids.append(download_id)
        assert len(_mesh_donwload_ids) == 1
        return _mesh_donwload_ids[0]

    meshfile_id = _find_single_download_id(RE_PATTERN_details_page_MESHFILE, html)
    meshfile_url = BIOMEDISA_DOWNLOAD_URL_TEMPLATE.format(id_=meshfile_id)

    imagefile_id = _find_single_download_id(RE_PATTERN_details_page_IMAGEFILE, html)
    imagefile_url = BIOMEDISA_DOWNLOAD_URL_TEMPLATE.format(id_=imagefile_id)

    return {"meshfile_url": meshfile_url, "imagefile_url": imagefile_url}


def fetch_and_parse_all_details_pages(
    parsed_index_page_df: pl.DataFrame, rewrite_existing: bool = False
) -> None:
    all_matches = list(parsed_index_page_df.iter_rows(named=True))
    for row in tqdm(all_matches, desc="Reading & parsing info pages"):
        specimen_dir = DATA_DIR / row["specimen_code"]
        specimen_dir.mkdir(parents=True, exist_ok=True)
        specimen_metadata_path = specimen_dir / "biomedisa_metadata.yaml"
        if not specimen_metadata_path.exists() or rewrite_existing:
            details_page_url = row["details_page_url"]
            download_links_dict = fetch_details_page_and_get_download_links(
                details_page_url
            )
            metadata_dict = {**row, **download_links_dict}
            with open(specimen_dir / "biomedisa_metadata.yaml", "w") as f:
                yaml.dump(metadata_dict, f)


def download_speciment(
    specimen_dir: Path,
    download_meshfile: bool = True,
    download_imagefile: bool = True,
    overwrite: bool = False,
) -> None:
    biomedisa_metdata_path = specimen_dir / "biomedisa_metadata.yaml"
    if not biomedisa_metdata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found for specimen {specimen_dir.name} at "
            f"{biomedisa_metdata_path}. Get download links first by parsing metadata."
        )
    with open(biomedisa_metdata_path, "r") as f:
        metadata_dict = yaml.safe_load(f)

    if download_meshfile:
        if len(list(specimen_dir.glob("*.stl"))) == 0 or overwrite:
            meshfile_url = metadata_dict["meshfile_url"]
            logger.info(
                f"Downloading mesh file for specimen {specimen_dir.name} "
                f"from {meshfile_url}"
            )
            wget.download(meshfile_url, out=specimen_dir.as_posix(), bar=None)
        else:
            logger.info(
                f"Mesh file already exists for specimen {specimen_dir.name}, skipping"
            )

    if download_imagefile:
        if len(list(specimen_dir.glob("*.tif"))) == 0 or overwrite:
            imagefile_url = metadata_dict["imagefile_url"]
            logger.info(
                f"Downloading image file for specimen {specimen_dir.name} "
                f"from {imagefile_url}"
            )
            wget.download(imagefile_url, out=specimen_dir.as_posix(), bar=None)
        else:
            logger.info(
                f"Image file already exists for specimen {specimen_dir.name}, skipping"
            )


if __name__ == "__main__":
    BASEDIR = get_bulkdata_dir() / "antscan"
    METADATA_DIR = BASEDIR / "metadata"
    DATA_DIR = BASEDIR / "dataset"
    BIOMEDISA_INDEX_PAGE_PATH = METADATA_DIR / "biomedisa_index_page.html"
    ANTSCAN_METADATA_CSV_PATH = METADATA_DIR / "antscan_metadata.csv"
    REFETCH_DOWNLOAD_LINKS = False

    DOWNLOAD_MESHFILES = True
    DOWNLOAD_IMAGEFILES = True
    DOWNLOAD_RANGE: tuple[int, int] | None = None
    REDOWNLOAD_DATA = False

    # ===== Get download links for scans and mesh files from BIOMEDISA webpages =====
    # Download index page and get per-specimen info page links
    parsed_index_page_df = fetch_index_page_and_get_details_pages(
        BIOMEDISA_INDEX_PAGE_URL, BIOMEDISA_INDEX_PAGE_PATH, REFETCH_DOWNLOAD_LINKS
    )
    metadata_df = pl.read_excel(METADATA_DIR / "antscan_metadata.xlsx")
    raise_on_inconsistent_metadata(parsed_index_page_df, metadata_df)

    # Download per-specimen info pages and get data download URLs
    fetch_and_parse_all_details_pages(parsed_index_page_df, REFETCH_DOWNLOAD_LINKS)

    # ===== Download data files =====
    all_specimen_codes = parsed_index_page_df["specimen_code"].to_list()
    if DOWNLOAD_RANGE is not None:
        logger.info(f"Limiting to specimens in range {DOWNLOAD_RANGE}")
        all_specimen_codes = all_specimen_codes[slice(*DOWNLOAD_RANGE)]
        logger.info(f"{len(all_specimen_codes)} specimens after applying range limit")
    start_time = time()
    for i, specimen_code in enumerate(all_specimen_codes):
        specimen_dir = DATA_DIR / specimen_code
        download_speciment(
            specimen_dir,
            download_meshfile=DOWNLOAD_MESHFILES,
            download_imagefile=DOWNLOAD_IMAGEFILES,
            overwrite=REDOWNLOAD_DATA,
        )
        logger.info(
            f"{i + 1}/{len(all_specimen_codes)} specimens downloaded "
            f"in {time() - start_time:.2f} seconds"
        )
    logger.success("All downloads completed.")
