import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full", app_title="Cell Image Processing")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # **Cell Image Processing Pipeline**

    ### Scripts that are ran during the pipeline (in order):
    1. Background removal
    2. Gamma correction
    3. Cell cropping
    4. Stitching cells
    5. Generating GIFs
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    from removeBackground import process_folder
    from gammaCorrection import process_images
    from cellCropping import divide_images
    from gifCreation import create_gif_from_stitched_images
    from stitchCells import stitch_all_modules

    # ------- Input Folder + Rows & Columns parameters --------
    input_folder   = "input"          # Enter raw image folder
    rows, columns  = 3, 12            # Enter grid layout of each module

    # derived folder names
    bg_removed     = f"{input_folder}_BackgroundRemoved"
    gamma_correct  = f"{input_folder}_GammaCorrected"
    cells_cropped  = f"{input_folder}_CellsCropped"
    cells_stitched = f"{input_folder}_CellsStitched"
    gif_res        = f"{input_folder}_GIFs"
    # ---------------------------------------------------------

    def run_pipeline(event):
        print("Step 1️: Running background removal...")
        process_folder(input_folder + "/", bg_removed + "/", threshold=10)

        print("\nStep 2️: Running gamma correction...")
        process_images(bg_removed + "/", gamma_correct + "/")

        print("\nStep 3️: Running cell cropping...")
        divide_images(gamma_correct + "/", cells_cropped + "/", rows=rows, columns=columns)

        print("\nStep 4: Stitching cells into full modules...")
        stitch_all_modules(cells_cropped + "/", cells_stitched + "/", rows=rows, columns=columns)

        print("\nStep 5: Creating GIFs from stitched modules...")
        create_gif_from_stitched_images(cells_stitched + "/", gif_res + "/", duration=500)

        print("\nFull pipeline completed!")

    # Create button to run the pipeline
    run_button = mo.ui.button(label="Run Image Processing Pipeline!", on_click=run_pipeline)

    run_button
    return (
        bg_removed,
        cells_cropped,
        cells_stitched,
        create_gif_from_stitched_images,
        divide_images,
        gamma_correct,
        gif_res,
        input_folder,
        mo,
        process_folder,
        process_images,
        stitch_all_modules,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 1 - Remove Background""")
    return


@app.cell
def _(bg_removed, input_folder, process_folder):
    process_folder(input_folder + "/", bg_removed + "/", threshold=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 2 - Gamma Correction""")
    return


@app.cell
def _(bg_removed, gamma_correct, process_images):
    process_images(bg_removed + "/", gamma_correct + "/")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 3 - Cell Cropping""")
    return


@app.cell
def _(cells_cropped, divide_images, gamma_correct):
    divide_images(gamma_correct + "/", cells_cropped + "/", rows=3, columns=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 4 - Stitching Cells""")
    return


@app.cell
def _(cells_cropped, cells_stitched, stitch_all_modules):
    stitch_all_modules(cells_cropped + "/", cells_stitched + "/", rows=3, columns=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 5 - Generating GIFs""")
    return


@app.cell
def _(cells_stitched, create_gif_from_stitched_images, gif_res):
    create_gif_from_stitched_images(cells_stitched + "/", gif_res + "/", duration=500)
    return


if __name__ == "__main__":
    app.run()
