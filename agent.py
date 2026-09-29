import os
import shutil
from pathlib import Path
import ollama

MODEL = "qwen3:8b"

# All file operations are restricted to this directory.
ROOT = Path(r"C:\Jamshed").resolve()
ROOT.mkdir(parents=True, exist_ok=True)

UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
CHUNK_SIZE = 1024 * 1024          # write 1 MB at a time so RAM use stays small
CONFIRM_ABOVE = 1024**3           # ask the user before creating files over 1 GB


def safe_path(path: str) -> Path:
    """Resolve a path and prevent access outside the workspace."""
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Access outside the workspace is prohibited.")
    return target


def create_file(path: str, content: str) -> str:
    """Create a new file with the specified text content."""
    target = safe_path(path)
    if target.exists():
        return "Error: File already exists."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Created: {target}"


def create_folder(path: str) -> str:
    """Create a new folder (including any missing parent folders) in the workspace."""
    target = safe_path(path)
    if target.exists():
        if target.is_dir():
            return "Error: Folder already exists."
        return "Error: A file with that name already exists."
    target.mkdir(parents=True, exist_ok=True)
    return f"Created folder: {target}"


def generate_file(
    path: str,
    size: float,
    unit: str = "MB",
    content_type: str = "text",
) -> str:
    """Generate a new file of an exact size filled with dummy data.

    Args:
        path: File path relative to the workspace, for example "test/big.bin".
        size: Size of the file as a number, for example 10 or 1.5.
        unit: Unit for the size. One of: B, KB, MB, GB, TB.
        content_type: Either "text" for readable filler text or "random" for random bytes.
    """
    unit = str(unit).strip().upper()
    content_type = str(content_type).strip().lower()

    if unit not in UNITS:
        return "Error: Unit must be one of B, KB, MB, GB, TB."
    if content_type not in ("text", "random"):
        return 'Error: content_type must be "text" or "random".'

    try:
        total_bytes = int(float(size) * UNITS[unit])
    except (TypeError, ValueError):
        return "Error: Size must be a number."
    if total_bytes <= 0:
        return "Error: Size must be greater than zero."

    target = safe_path(path)
    if target.exists():
        return "Error: File already exists."

    free = shutil.disk_usage(ROOT).free
    if total_bytes >= free:
        return (
            f"Error: Not enough disk space. Requested {total_bytes:,} bytes, "
            f"only {free:,} bytes free."
        )

    if total_bytes >= CONFIRM_ABOVE:
        answer = input(
            f"Confirm creating a {total_bytes / 1024**3:.2f} GB file at {target}? (yes/no): "
        )
        if answer.strip().lower() != "yes":
            return "File generation cancelled."

    target.parent.mkdir(parents=True, exist_ok=True)

    # Pre-build one chunk of readable text (ASCII, so 1 character = 1 byte).
    text_line = b"The quick brown fox jumps over the lazy dog. 0123456789\n"
    text_chunk = (text_line * (CHUNK_SIZE // len(text_line) + 1))[:CHUNK_SIZE]

    remaining = total_bytes
    try:
        with open(target, "wb") as f:
            while remaining > 0:
                n = min(CHUNK_SIZE, remaining)
                data = text_chunk[:n] if content_type == "text" else os.urandom(n)
                f.write(data)
                remaining -= n
    except BaseException:
        # Remove the half-written file if something goes wrong or you press Ctrl+C.
        target.unlink(missing_ok=True)
        raise

    return f"Generated: {target} ({total_bytes:,} bytes)"


def rename_file(old_path: str, new_path: str) -> str:
    """Rename or move a file within the workspace."""
    source = safe_path(old_path)
    destination = safe_path(new_path)

    if not source.is_file():
        return "Error: Source file does not exist."
    if destination.exists():
        return "Error: Destination already exists."

    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    return f"Renamed to: {destination}"


def delete_file(path: str) -> str:
    """Delete a file after obtaining user confirmation."""
    target = safe_path(path)

    if not target.is_file():
        return "Error: File does not exist."

    answer = input(f"Confirm deletion of {target}? (yes/no): ")
    if answer.strip().lower() != "yes":
        return "Deletion cancelled."

    target.unlink()
    return f"Deleted: {target}"


def list_files(path: str = ".") -> str:
    """List files and directories in a workspace folder."""
    target = safe_path(path)
    if not target.is_dir():
        return "Error: Directory does not exist."

    return "\n".join(
        ("[DIR] " if p.is_dir() else "[FILE] ") + p.name
        for p in sorted(target.iterdir())
    ) or "Directory is empty."


TOOLS = [create_file, create_folder, generate_file, rename_file, delete_file, list_files]

SYSTEM_PROMPT = """
You are a local file-management assistant.
Use the available tools to create files, create folders, generate files of a given size, rename, delete and list files.
Use generate_file when the user asks for a file of a specific size (for example "a 50 MB file").
All paths are relative to the provided workspace.
Never claim that an operation succeeded unless its tool confirms it.
Ask the user for missing file names or content.
Do not attempt to bypass tool restrictions.
"""


def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Local Qwen3 File Agent")
    print("Workspace:", ROOT)
    print("Type 'exit' to quit.")

    while True:
        prompt = input("\nEnter file details: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})

        while True:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )

            messages.append(response.message)

            if not response.message.tool_calls:
                print("\nAgent:", response.message.content)
                break

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments

                tool = next(
                    (t for t in TOOLS if t.__name__ == name),
                    None
                )

                if tool is None:
                    result = "Error: Unknown tool."
                else:
                    try:
                        result = tool(**args)
                    except Exception as exc:
                        result = f"Error: {exc}"

                print(f"\n[Tool: {name}]")
                print(result)

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                })


if __name__ == "__main__":
    run_agent()