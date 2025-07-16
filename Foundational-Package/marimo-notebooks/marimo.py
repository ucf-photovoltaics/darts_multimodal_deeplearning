import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full", app_title="Cell Image Processing")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # **Cell Image Processing Pipeline**

    The marimo notebook reads raw EL images from an **`input/`** folder located at
    `Foundational‑Package/input/` and runs the following scripts (in order):

    ### Scripts that are ran during the pipeline (in order):
    1. Background removal
    2. Gamma correction
    3. Cell cropping
    4. Stitching cells
    5. Generating GIFs
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[1]                  # Foundational-Package
    img_proc_dir = project_root / "img-processing"   # img-processing folder

    if img_proc_dir.as_posix() not in sys.path:
        sys.path.append(img_proc_dir.as_posix())

    from removeBackground import process_folder
    from gammaCorrection import process_images
    from cellCropping import divide_images
    from gifCreation import create_gif_from_stitched_images
    from stitchCells import stitch_all_modules

    # ------- Input Folder + Rows & Columns parameters --------
    rows, columns  = 3, 12                              # Enter grid layout of each module
    
    input_dir      = project_root / "input"             # Input folder containing images to process
    output_dir     = project_root / "output"            # Output folder for processed image

    bg_removed     = output_dir / "BackgroundRemoved"
    gamma_correct  = output_dir / "GammaCorrected"
    cells_cropped  = output_dir / "CellsCropped"
    cells_stitched = output_dir / "CellsStitched"
    gif_res        = output_dir / "GIFs"

    # Ensure output folders exist before the buttons are pressed
    for p in [output_dir, bg_removed, gamma_correct, cells_cropped, cells_stitched, gif_res]:
        p.mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------

    def run_pipeline(event):
        print("Step 1️: Running background removal...")
        process_folder(str(input_dir) + "/", str(bg_removed) + "/", threshold=10)

        print("\nStep 2️: Running gamma correction...")
        process_images(str(bg_removed) + "/", str(gamma_correct) + "/")

        print("\nStep 3️: Running cell cropping...")
        divide_images(str(gamma_correct) + "/", str(cells_cropped) + "/", rows=rows, columns=columns)

        print("\nStep 4: Stitching cells into full modules...")
        stitch_all_modules(str(cells_cropped) + "/", str(cells_stitched) + "/", rows=rows, columns=columns)

        print("\nStep 5: Creating GIFs from stitched modules...")
        create_gif_from_stitched_images(str(cells_stitched) + "/", str(gif_res) + "/", duration=500)

        print("\nFull pipeline completed!")

    # Create button to run the pipeline
    run_button = mo.ui.button(label="Run Image Processing Pipeline!", on_click=run_pipeline)

    run_button
    return (
        rows,
        columns,
        input_dir,
        output_dir,
        bg_removed,
        gamma_correct,
        cells_cropped,
        cells_stitched,
        gif_res,
        run_button,
        divide_images,
        process_folder,
        process_images,
        stitch_all_modules,
        create_gif_from_stitched_images,
        mo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Run scripts individually below:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 1 - Remove Background""")
    return


@app.cell
def _(bg_removed, input_dir, process_folder):
    process_folder(str(input_dir) + "/", str(bg_removed) + "/", threshold=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 2 - Gamma Correction""")
    return


@app.cell
def _(bg_removed, gamma_correct, process_images):
    process_images(str(bg_removed) + "/", str(gamma_correct) + "/")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 3 - Cell Cropping""")
    return


@app.cell
def _(cells_cropped, columns, divide_images, gamma_correct, rows):
    divide_images(str(gamma_correct) + "/", str(cells_cropped) + "/", rows=rows, columns=columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 4 - Stitching Cells""")
    return


@app.cell
def _(cells_cropped, cells_stitched, columns, rows, stitch_all_modules):
    stitch_all_modules(str(cells_cropped) + "/", str(cells_stitched) + "/", rows=rows, columns=columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 5 - Generating GIFs""")
    return


@app.cell
def _(cells_stitched, create_gif_from_stitched_images, gif_res):
    create_gif_from_stitched_images(str(cells_stitched) + "/", str(gif_res) + "/", duration=500)
    return


if __name__ == "__main__":
    app.run()
