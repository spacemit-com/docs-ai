---
sidebar_position: 12
---

# File2MD

**File2MD** is an on-device document-to-Markdown application for K3 systems running Bianbu Desktop. It turns documents, images, and web pages into structured Markdown while preserving useful elements such as headings, tables, formulas, flowcharts, and extracted images.

The desktop icon opens File2MD in a dedicated Chromium application window. The interface is served by a local service at `127.0.0.1:8070`; documents stay on the device throughout local-file conversion.

## Key Features

- **Broad Format Support**: Convert PDF, Word, PowerPoint, Excel, text, and common image formats from one interface.
- **Structured Document Parsing**: Detect page layout, text, formulas, tables, flowcharts, and image captions.
- **On-Device AI Inference**: Run MinerU with SpacemiT ORT on the K3 platform without uploading local documents to a cloud service.
- **Original and Markdown Comparison**: Review the source document and rendered Markdown side by side.
- **Reusable Output**: Switch between rendered and source views, copy Markdown, or download a ZIP containing the `.md` file and its `images/` resources.
- **Queue and History Management**: Add multiple files, follow per-job progress, and reopen completed conversions from the history list.
- **Web Page Conversion**: Submit a public URL and convert its readable content to Markdown when network access is available.

## Platform Support

| Platform & OS | Supported |
|---|---|
| K1 Buildroot | No |
| K1 OpenHarmony | No |
| K1 Bianbu LXQt/GNOME | No |
| K3 Buildroot | No |
| K3 OpenHarmony | No |
| K3 Bianbu LXQt/GNOME | Yes |

## Technical Architecture

### Application Stack

- **Desktop Entry**: `file2md-launcher` opens Chromium in application mode.
- **Frontend**: Static HTML, CSS, and JavaScript served locally.
- **Backend**: FastAPI and Uvicorn on Python 3.14.
- **Document Engine**: MinerU `0.1.0`.
- **AI Runtime**: `python3-spacemit-ort (>= 2.0.6)` for K3 inference.
- **Office Preview**: LibreOffice in headless mode.
- **PDF Preview**: PDFium-based page rendering.
- **Markdown Rendering**: Marked and DOMPurify, with KaTeX and Mermaid support.

### Runtime Dependencies

The Debian package declares the system dependencies required by File2MD. In particular, `apt` resolves `python3-spacemit-ort (>= 2.0.6)` during installation. The package also deploys File2MD's private Python environment, frontend, service definition, and desktop launcher.

The model files are installed separately from the application code under `/var/lib/file2md/models/`. On the first installation, the package downloads and verifies approximately 1.6 GB of model data, so the device must have network access and sufficient free storage.

### System Architecture Diagram

![File2MD system architecture](../static/file2md_en-framework.png)

### Processing Workflow

1. **Submit Input**: Drop one or more local files into the interface, choose files from disk, or enter a web URL.
2. **Prepare the Source**: Office files are converted for preview, while PDF and image inputs are normalized for document analysis.
3. **Analyze the Document**: MinerU performs layout analysis, OCR, formula and table recognition, and any enabled flowchart or image-caption processing.
4. **Run Local Inference**: SpacemiT ORT executes the required models on the K3 platform.
5. **Build the Result**: File2MD writes Markdown and image resources, then prepares the original preview and downloadable ZIP.
6. **Review or Reopen**: View the result immediately or reopen it later from conversion history.

## Installation

Install File2MD from the Bianbu package repository:

```bash
sudo apt update
sudo apt install file2md
```

The installation enables and starts `file2md.service`. Model download and verification may take several minutes on the first installation. Do not close the terminal or disconnect the device while this step is running.

## Quick Start

### 1. Launch File2MD

Open the system application menu, enter **File2MD** in the search field, and click the **File2MD** application shown in the results.

![Search for and launch File2MD from the system application menu](../static/file2md_en-launch.png)

The dedicated desktop window and a regular browser tab use the same local web interface and backend. The desktop launcher removes normal browser navigation controls to provide a focused application window; it does not send the document to a remote website.

> The screenshots show a Chinese-language interface. The highlighted controls and workflow described below are the same.

### 2. Add a File or URL

Check the model indicator in the upper-right corner and wait until it reports **Ready**. The first launch after installation may take longer while the runtime initializes.

For a local document, drag and drop it into the upload area highlighted by the upper arrow, or click that area to choose a file. For a web page, enter its address in the URL field highlighted by the lower arrow. Each selected input is added to the file queue. Review the queue, then click **Start Conversion** in the highlighted box.

![Add a local file or URL, review the queue, and start conversion](../static/file2md_en-input.png)

### 3. Configure Parsing

The highlighted progress area contains six stage indicators: **Layout**, **OCR**, **Formula**, **Table**, **Flowchart**, and **Image Caption**. They show which parts of document analysis are pending, running, or complete.

The highlighted parsing-options area provides two switches and an inference-thread selector:

- **Flowchart recognition** converts recognized flowcharts to Mermaid markup.
- **Document image text recognition** recognizes text inside document images and writes it to the result as annotations.
- **EP inference threads** controls the number of threads used for model inference. Higher values generally use more processor and memory resources, so select a value appropriate for the device load and document size. The **6-core** value in the screenshot is an example setting, not a fixed requirement.

Enable the two optional recognition switches only when the source needs them.

![Monitor the six parsing stages and configure optional recognition and EP threads](../static/file2md_en-options.png)

### 4. Review the Result

When conversion completes, File2MD displays the original document and rendered Markdown side by side. The highlighted list on the left is the conversion history; select an item to reopen it. The upper-right result badge, indicated by the upper arrow, reports the inference duration for the selected job.

Use the preview/source tabs to inspect the generated Markdown. The **Copy Markdown** button is visible below the result, and **Download ZIP**, indicated by the lower-right arrow, saves the `.md` file together with its extracted image resources.

![Review history and inference duration, then copy Markdown or download the ZIP](../static/file2md_en-result.png)

### 5. Open a Built-in Example

The **Examples** area highlighted in the screenshot contains ready-made results that require no file upload. Its cards demonstrate conversion of a web article with code and images, formula and table recognition, retained extracted images, and preservation of document structure. Select a card to inspect that example, or click **View All** at the upper right of the area to browse the full set.

![Open a built-in example or view the complete example collection](../static/file2md_en-examples.png)

## Supported Input Formats

| Category | Formats |
|---|---|
| Documents | `PDF`, `PPT`, `PPTX`, `DOC`, `DOCX`, `XLS`, `XLSX`, `TXT` |
| Images | `PNG`, `JPG`, `JPEG`, `JP2`, `WEBP`, `GIF`, `BMP`, `TIFF` |
| Web | `HTTP` or `HTTPS` URL |

## Document Conversion

### PDF and Images

PDF pages and image files enter the document-analysis pipeline directly. File2MD can recover paragraphs and headings, recognize text with OCR, preserve tables and formulas, and extract embedded images. PDF previews use server-side page rendering. Completed PDF and image jobs can both be previewed again from history.

For scanned documents, processing time depends on page count, resolution, enabled recognition options, and available memory. Very large inputs should be divided into smaller documents when possible.

### Word, PowerPoint, and Spreadsheets

File2MD uses LibreOffice in headless mode to prepare Office documents for preview and parsing. After conversion, Word, PowerPoint, and spreadsheet source content can be inspected in the original pane, while the generated Markdown appears alongside it.

Complex slide animations, macros, embedded media, or spreadsheet formulas may not have a direct Markdown equivalent. Review the output before using it as a definitive replacement for the source document.

### Web URLs

URL conversion fetches the target page over the network and converts its readable content. It is the only conversion path that requires external network access; local file conversion remains on-device.

The current URL workflow does not retain the original HTML after processing. The Markdown result remains available in history, but the original web page cannot be reopened in the source-preview pane from that historical job.

## Result and History Management

### Markdown Preview

The rendered view supports headings, lists, tables, extracted images, KaTeX formulas, and Mermaid diagrams. Switch to the source view whenever the exact Markdown syntax is needed.

### Original Preview

The original pane is a document preview, independent of the generated Markdown. It remains available when completed local-file jobs are reopened from history, provided the corresponding original and preview data have not been removed from File2MD's data directory.

### ZIP Download

The ZIP result contains the generated `.md` file and an `images/` directory when the conversion produces image assets. Keep these paths together so image references in the Markdown continue to resolve.

### Conversion History

Recent jobs are listed in the left sidebar. A completed local-file job can reopen its original preview and Markdown result, while a URL job reopens the Markdown result only; the inference duration appears in the result header. Removing File2MD's persistent data manually also removes the history or the files required by its previews.

### Built-in Examples

Built-in examples are separate from conversion history and can be opened without uploading a source file. They provide quick reference results for web content, code and images, formulas and tables, extracted-image retention, and document-structure retention. Use **View All** in the Examples area to open the complete collection.

## Local Runtime and Data

File2MD separates immutable application files, configuration, persistent models and results, and disposable cache data:

| Path | Purpose |
|---|---|
| `/opt/file2md/frontend/` | Static web interface |
| `/opt/file2md/backend-venv/` | Private Python runtime and backend packages |
| `/etc/file2md/` | System configuration |
| `/var/lib/file2md/models/` | Persistent model files |
| `/var/lib/file2md/data/jobs/` | Job metadata and history |
| `/var/lib/file2md/data/originals/` | Retained source files used for preview |
| `/var/lib/file2md/data/outputs/` | Generated Markdown and related output |
| `/var/cache/file2md/` | Rebuildable preview and processing cache |

These system directories are created and managed by the Debian package. They are not expected to exist in the source repository.

## Service Status and Troubleshooting

Check whether the service is running:

```bash
systemctl status file2md.service
```

Follow backend logs while reproducing a conversion issue:

```bash
journalctl -u file2md.service -f
```

Confirm that the local service is listening on port `8070`:

```bash
ss -ltn | grep 8070
```

If the desktop icon opens an error page, first verify the service status and port. If a conversion fails because of insufficient memory, close other memory-intensive applications, disable optional recognition stages that the document does not need, and retry with a smaller input. Preserve the relevant service log when reporting a repeatable failure.
