"""GUI application for dive color correction."""

from pathlib import Path

import PySimpleGUI as sg  # noqa: N813

from dive_color_corrector.core.models import SESR_AVAILABLE
from dive_color_corrector.core.processing.image import correct_image
from dive_color_corrector.core.processing.video import analyze_video, process_video
from dive_color_corrector.core.schemas import VideoData
from dive_color_corrector.core.utils.constants import (
    IMAGE_FORMATS,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    VIDEO_FORMATS,
)

# Window settings
WINDOW_SIZE = (1100, 750)


def create_window() -> sg.Window:
    """Create and return the main application window."""
    sg.theme("DarkBlue")
    sg.set_options(font=("Helvetica", 11))

    all_formats = list(IMAGE_FORMATS) + list(VIDEO_FORMATS)
    file_types = (
        ("All Supported", " ".join(f"*{ext}" for ext in all_formats)),
        ("Images", " ".join(f"*{ext}" for ext in IMAGE_FORMATS)),
        ("Videos", " ".join(f"*{ext}" for ext in VIDEO_FORMATS)),
    )

    # Left column: file selection and options
    left_col = [
        [sg.Text("Input Files", font=("Helvetica", 12, "bold"))],
        [
            sg.Input(key="-FILES_INPUT-", enable_events=True, visible=False),
            sg.FilesBrowse(
                "Select Files",
                file_types=file_types,
                key="-BROWSE-",
                target="-FILES_INPUT-",
            ),
            sg.Button("Clear", key="-CLEAR-", size=(8, 1)),
        ],
        [
            sg.Listbox(
                values=[],
                size=(35, 12),
                key="-FILE_LIST-",
                enable_events=True,
                horizontal_scroll=True,
                select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
            )
        ],
        [sg.Text("Drag files here or use Select Files button", font=("Helvetica", 9, "italic"))],
        [sg.HSeparator()],
        [sg.Text("Options", font=("Helvetica", 12, "bold"))],
        [
            sg.Checkbox(
                "Use Deep Learning (SESR)" + ("" if SESR_AVAILABLE else " [Not Available]"),
                key="-USE_DEEP-",
                default=False,
                disabled=not SESR_AVAILABLE,
                tooltip="Neural network enhancement - slower but often better quality"
                + ("" if SESR_AVAILABLE else " (requires onnxruntime)"),
            )
        ],
        [sg.HSeparator()],
        [sg.Text("Output Folder:", size=(11, 1))],
        [
            sg.Input(
                default_text=str(Path.home() / "Pictures" / "Corrected"),
                key="-OUTPUT-",
                size=(28, 1),
            ),
            sg.FolderBrowse(size=(7, 1)),
        ],
        [sg.VPush()],
        [
            sg.Button(
                "Process All",
                key="-PROCESS-",
                disabled=True,
                size=(15, 1),
                button_color=("white", "#FF6B35"),
            )
        ],
    ]

    # Right column: preview with before/after labels
    right_col = [
        [
            sg.Column(
                [[sg.Text("Original", font=("Helvetica", 10))]],
                element_justification="center",
                expand_x=True,
            ),
            sg.Column(
                [[sg.Text("Corrected", font=("Helvetica", 10))]],
                element_justification="center",
                expand_x=True,
            ),
        ],
        [sg.Image(key="-PREVIEW-", size=(PREVIEW_WIDTH, PREVIEW_HEIGHT))],
        [sg.HSeparator()],
        [
            sg.ProgressBar(
                100,
                orientation="h",
                size=(45, 20),
                key="-PROGRESS-",
                bar_color=("#FF6B35", "#2C3E50"),
            )
        ],
        [
            sg.Text(
                "Ready",
                key="-STATUS-",
                size=(55, 1),
                justification="center",
                font=("Helvetica", 10),
            )
        ],
    ]

    layout = [
        [
            sg.Text(
                "Dive Color Corrector",
                font=("Helvetica", 18, "bold"),
                justification="center",
                expand_x=True,
                pad=(0, 10),
            )
        ],
        [sg.HSeparator()],
        [
            sg.Column(left_col, vertical_alignment="top", pad=(10, 10)),
            sg.VSeparator(),
            sg.Column(right_col, vertical_alignment="top", expand_x=True, pad=(10, 10)),
        ],
    ]

    return sg.Window(
        "Dive Color Corrector",
        layout,
        size=WINDOW_SIZE,
        finalize=True,
        enable_close_attempted_event=True,
        enable_drag_and_drop=True,
    )


def valid_file(path: str) -> bool:
    """Check if file is a valid image or video."""
    return path.lower().endswith(tuple(IMAGE_FORMATS) + tuple(VIDEO_FORMATS))


def get_files(filepaths: str) -> list[str]:
    """Get list of valid files from filepaths."""
    return [f for f in filepaths.split(";") if valid_file(f)]


def process_files(window: sg.Window, files: list[str], output_folder: str) -> None:
    """Process the selected files."""
    total_files = len(files)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    use_deep = window["-USE_DEEP-"].get()

    for i, file in enumerate(files, 1):
        filename = Path(file).name
        output_file = output_path / f"corrected_{filename}"
        base_progress = ((i - 1) * 100) // total_files

        window["-STATUS-"].update(f"[{i}/{total_files}] Processing {filename}...")
        window["-PROGRESS-"].update(base_progress)
        window.refresh()

        try:
            if file.lower().endswith(tuple(IMAGE_FORMATS)):
                preview_data = correct_image(file, str(output_file), use_deep=use_deep)
                window["-PREVIEW-"].update(data=preview_data)
                window["-PROGRESS-"].update((i * 100) // total_files)
            else:
                video_data = None
                for data in analyze_video(file, str(output_file)):
                    if isinstance(data, VideoData):
                        video_data = data
                    else:
                        window["-STATUS-"].update(
                            f"[{i}/{total_files}] Analyzing {filename}: {data} frames"
                        )
                        window.refresh()

                if video_data is None:
                    window["-STATUS-"].update(f"Error: Failed to analyze {filename}")
                    continue

                for percent, preview in process_video(
                    video_data, yield_preview=True, use_deep=use_deep
                ):
                    if preview:
                        window["-PREVIEW-"].update(data=preview)
                    if percent is not None:
                        file_progress = base_progress + (percent * (100 // total_files)) // 100
                        window["-PROGRESS-"].update(int(file_progress))
                        window["-STATUS-"].update(
                            f"[{i}/{total_files}] Encoding {filename}: {int(percent)}%"
                        )
                    window.refresh()
        except Exception as e:
            window["-STATUS-"].update(f"Error: {filename} - {e!s}")
            continue

    window["-STATUS-"].update(f"Done! {total_files} file(s) saved to {output_folder}")
    window["-PROGRESS-"].update(100)


def run_gui() -> None:
    """Run the GUI application."""
    window = create_window()
    selected_files: list[str] = []

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, sg.WINDOW_CLOSE_ATTEMPTED_EVENT):
            break

        if event == "-FILES_INPUT-":
            new_files = get_files(values["-FILES_INPUT-"])
            if new_files:
                selected_files = list(dict.fromkeys(selected_files + new_files))
                display_names = [Path(f).name for f in selected_files]
                window["-FILE_LIST-"].update(display_names)
                window["-PROCESS-"].update(disabled=False)
                window["-STATUS-"].update(f"{len(selected_files)} file(s) selected")

        if event == sg.DND_FILES:
            dropped_files = get_files(",".join(values[event]))
            if dropped_files:
                selected_files = list(dict.fromkeys(selected_files + dropped_files))
                display_names = [Path(f).name for f in selected_files]
                window["-FILE_LIST-"].update(display_names)
                window["-PROCESS-"].update(disabled=False)
                window["-STATUS-"].update(f"{len(selected_files)} file(s) selected")

        if event == "-CLEAR-":
            selected_files = []
            window["-FILE_LIST-"].update([])
            window["-PROCESS-"].update(disabled=True)
            window["-PREVIEW-"].update(data=None)
            window["-STATUS-"].update("Ready")
            window["-PROGRESS-"].update(0)

        if event == "-FILE_LIST-":
            selection_indices = window["-FILE_LIST-"].get_indexes()
            if selection_indices and selected_files:
                selected_index = selection_indices[0]
                if selected_index < len(selected_files):
                    file_path = selected_files[selected_index]
                    if file_path.lower().endswith(tuple(IMAGE_FORMATS)):
                        window["-STATUS-"].update(
                            f"Generating preview for {Path(file_path).name}..."
                        )
                        window.refresh()
                        try:
                            preview_data = correct_image(
                                file_path, None, use_deep=values["-USE_DEEP-"]
                            )
                            window["-PREVIEW-"].update(data=preview_data)
                            window["-STATUS-"].update(f"Preview: {Path(file_path).name}")
                        except Exception as e:
                            window["-STATUS-"].update(f"Preview error: {e!s}")

        if event == "-PROCESS-":
            output_folder = values["-OUTPUT-"]
            process_files(window, selected_files, output_folder)

    window.close()
