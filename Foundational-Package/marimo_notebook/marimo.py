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

    ## Then choose a Defect Segmentation script to run on the cropped cell images generated as part of the pipeline.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sys, importlib
    from pathlib import Path

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[1]                  # Foundational-Package
    img_proc_dir = project_root / "img_processing"          # img_processing folder

    if img_proc_dir.as_posix() not in sys.path:
        sys.path.append(img_proc_dir.as_posix())

    from removeBackground import process_folder as rmb
    from gammaCorrection import process_images as gc
    from cellCropping import divide_images as crp
    from stitchCells import stitch_all_modules as stc
    from gifCreation import create_gif_from_stitched_images as gif

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
        # img processing scripts
        print("Step 1️: Running background removal...")
        rmb(str(input_dir) + "/", str(bg_removed) + "/", threshold=10)

        print("\nStep 2️: Running gamma correction...")
        gc(str(bg_removed) + "/", str(gamma_correct) + "/")

        print("\nStep 3️: Running cell cropping...")
        crp(str(gamma_correct) + "/", str(cells_cropped) + "/", rows=rows, columns=columns)

        print("\nStep 4: Stitching cells into full modules...")
        stc(str(cells_cropped) + "/", str(cells_stitched) + "/", rows=rows, columns=columns)

        print("\nStep 5: Creating GIFs from stitched modules...")
        gif(str(cells_stitched) + "/", str(gif_res) + "/", duration=500)

        print("\nFull image processing pipeline completed!")

    # Create button to run the pipeline
    run_button = mo.ui.button(label="Run Image Processing Pipeline!", on_click=run_pipeline)

    run_button
    return (
        bg_removed,
        cells_cropped,
        cells_stitched,
        columns,
        crp,
        gamma_correct,
        gc,
        gif,
        gif_res,
        input_dir,
        mo,
        rmb,
        rows,
        stc,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Run image processing scripts individually below:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 1 - Remove Background""")
    return


@app.cell
def _(bg_removed, input_dir, rmb):
    rmb(str(input_dir) + "/", str(bg_removed) + "/", threshold=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 2 - Gamma Correction""")
    return


@app.cell
def _(bg_removed, gamma_correct, gc):
    gc(str(bg_removed) + "/", str(gamma_correct) + "/")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 3 - Cell Cropping""")
    return


@app.cell
def _(cells_cropped, columns, crp, gamma_correct, rows):
    crp(str(gamma_correct) + "/", str(cells_cropped) + "/", rows=rows, columns=columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 4 - Stitching Cells""")
    return


@app.cell
def _(cells_cropped, cells_stitched, columns, rows, stc):
    stc(str(cells_cropped) + "/", str(cells_stitched) + "/", rows=rows, columns=columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 5 - Generating GIFs""")
    return


@app.cell
def _(cells_stitched, gif, gif_res):
    gif(str(cells_stitched) + "/", str(gif_res) + "/", duration=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Run the chosen defect segmentation script below:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## EL Defect Segmentation""")
    return


@app.cell
def _(mo):
    def _():
        import sys
        import importlib
        from pathlib import Path

        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]            # Foundational‑Package/
        if project_root.as_posix() not in sys.path:
            sys.path.append(project_root.as_posix())

        if 'cell_cropping' not in sys.modules:
            cellCropping_mod = importlib.import_module('cellCropping')
            sys.modules['cell_cropping'] = cellCropping_mod

        # We import inside the callback so the button remains clickable
        def run_segmentation(event):
            from defect_segmentation import ELDefectSegmentation as seg  # noqa: F401
            import importlib
            importlib.reload(seg)   # allows re‑running without restarting the app

        seg_button = mo.ui.button(label="Run EL defect segmentation!", on_click=run_segmentation)
        return seg_button
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Pixel‑level defect counting & visuals""")
    return


@app.cell
def _(mo):
    def _():
        import sys, importlib
        from pathlib import Path

        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]
        if project_root.as_posix() not in sys.path:
            sys.path.append(project_root.as_posix())

        if 'cell_cropping' not in sys.modules:
            cellCropping_mod = importlib.import_module('cellCropping')
            sys.modules['cell_cropping'] = cellCropping_mod

        # Callback to run the pixel‑count script
        def run_pixel_count(event):
            from defect_segmentation import segmentationPixelCount as spc  # noqa: F401
            import importlib
            importlib.reload(spc)   # allows re‑running without restarting the app

        pixel_button = mo.ui.button(label="Run pixel‑level analysis!", on_click=run_pixel_count)
        return pixel_button
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mask separator""")
    return


@app.cell
def _(mo):
    def _():
        import sys, importlib
        from pathlib import Path

        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]
        if project_root.as_posix() not in sys.path:
            sys.path.append(project_root.as_posix())

        if 'cell_cropping' not in sys.modules:
            sys.modules['cell_cropping'] = importlib.import_module('cellCropping')

        def run_mask_separator(event):
            from defect_segmentation import maskSeparator2 as ms  # noqa: F401
            import importlib
            importlib.reload(ms)

        mask_button = mo.ui.button(label="Run mask separator!", on_click=run_mask_separator)
        return mask_button
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Build cell‑level defect dataframe""")
    return


@app.cell
def _(mo):
    def _():
        import sys, importlib
        from pathlib import Path

        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]
        if project_root.as_posix() not in sys.path:
            sys.path.append(project_root.as_posix())

        def run_dataframe(event):
            from defect_segmentation import defectDataframeV1 as dfv  # noqa: F401
            import importlib
            importlib.reload(dfv)

        defect_button = mo.ui.button(label="Generate defect dataframe!", on_click=run_dataframe)
        return defect_button
    _()
    return


if __name__ == "__main__":
    app.run()
