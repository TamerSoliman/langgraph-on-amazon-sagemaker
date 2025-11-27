"""
File System Tool Example

Demonstrates how to create custom tools that interact with the file system.

For AI/ML Scientists:
This allows your model to read data files, configs, logs, or write outputs.
Think of it as giving your model I/O capabilities beyond just text processing.

SECURITY WARNING:
File system access is powerful but dangerous! This example implements
sandboxing (restricted to specific directories) to prevent security issues.
"""

import os
import pathlib
from typing import List
from langchain.tools import tool


# =============================================================================
# CONFIGURATION
# =============================================================================

# Allowed directories for read/write operations
# For AI/ML Scientists: NEVER allow unrestricted file access! Always sandbox
# to specific directories. This is like setting permissions in Unix.

ALLOWED_READ_DIR = os.getenv("ALLOWED_FILE_DIR", "/tmp/safe_files")
ALLOWED_WRITE_DIR = os.getenv("ALLOWED_WRITE_DIR", "/tmp/safe_output")

# Maximum file size to read (prevent memory issues with huge files)
MAX_FILE_SIZE_MB = 10  # 10 MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Allowed file extensions (security: prevent reading sensitive files)
ALLOWED_EXTENSIONS = ['.txt', '.md', '.json', '.csv', '.log', '.yaml', '.yml']


# =============================================================================
# SECURITY HELPERS
# =============================================================================

def is_path_safe(filepath: str, allowed_dir: str) -> bool:
    """
    Checks if filepath is within allowed directory (prevents path traversal attacks).

    For AI/ML Scientists:
    Path traversal attack example:
    - User inputs: "../../../etc/passwd"
    - Without validation, this could read sensitive system files!
    - This function prevents that by checking the resolved path.

    Args:
        filepath: Path to check
        allowed_dir: Directory that access is restricted to

    Returns:
        True if path is safe, False otherwise
    """

    # Resolve to absolute path (handles .., symlinks, etc.)
    try:
        abs_filepath = os.path.abspath(filepath)
        abs_allowed = os.path.abspath(allowed_dir)

        # Check if filepath is inside allowed directory
        return abs_filepath.startswith(abs_allowed)

    except Exception:
        return False


def is_extension_allowed(filepath: str) -> bool:
    """Checks if file extension is in allowed list."""

    ext = pathlib.Path(filepath).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def ensure_directory_exists(directory: str):
    """Creates directory if it doesn't exist."""

    os.makedirs(directory, exist_ok=True)


# =============================================================================
# READ TOOLS
# =============================================================================

@tool
def read_file(filename: str) -> str:
    """
    Read the contents of a text file.

    Use this tool when the user asks to:
    - Read a file
    - View file contents
    - Summarize a document
    - Analyze logs

    Examples:
    - "What's in the config.json file?"
    - "Read the contents of report.txt"
    - "Summarize the log file from yesterday"

    Args:
        filename: Name of file to read (must be in allowed directory)

    Returns:
        File contents or error message

    For AI/ML Scientists:
    This is like a data loader that reads from disk instead of a database.
    We read the entire file into memory, so there's a size limit to prevent
    OOM errors.
    """

    print(f"[FILE TOOL] Reading file: {filename}")

    # Build full path
    filepath = os.path.join(ALLOWED_READ_DIR, filename)

    # Security checks
    if not is_path_safe(filepath, ALLOWED_READ_DIR):
        return f"Error: Access denied. File must be in {ALLOWED_READ_DIR}"

    if not is_extension_allowed(filepath):
        return f"Error: File type not allowed. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"

    try:
        # Check file exists
        if not os.path.exists(filepath):
            # List available files to help user
            available = list_files.invoke("")
            return f"Error: File '{filename}' not found.\n\n{available}"

        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE_BYTES:
            return f"Error: File too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB."

        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            contents = f.read()

        print(f"[FILE TOOL] ✓ Read {len(contents)} characters")

        # Return with metadata
        return f"""File: {filename}
Size: {len(contents)} characters
---
{contents}
"""

    except UnicodeDecodeError:
        return f"Error: File '{filename}' is not a text file (binary content detected)"

    except PermissionError:
        return f"Error: Permission denied reading '{filename}'"

    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def list_files(directory: str = "") -> str:
    """
    List files in a directory.

    Use this tool when the user asks to:
    - See available files
    - List files in a directory
    - Find a file

    Args:
        directory: Subdirectory to list (empty string for root)

    Returns:
        List of files and directories

    For AI/ML Scientists:
    This is like exploring a dataset directory to see what files are available
    before loading them for training.
    """

    print(f"[FILE TOOL] Listing files in: {directory or 'root'}")

    # Build full path
    if directory:
        dirpath = os.path.join(ALLOWED_READ_DIR, directory)
    else:
        dirpath = ALLOWED_READ_DIR

    # Security check
    if not is_path_safe(dirpath, ALLOWED_READ_DIR):
        return f"Error: Access denied. Must be within {ALLOWED_READ_DIR}"

    try:
        # Ensure directory exists
        if not os.path.exists(dirpath):
            return f"Directory not found: {directory or 'root'}"

        if not os.path.isdir(dirpath):
            return f"Error: '{directory}' is not a directory"

        # List contents
        items = os.listdir(dirpath)

        if not items:
            return f"Directory '{directory or 'root'}' is empty"

        # Separate files and directories
        files = []
        dirs = []

        for item in sorted(items):
            item_path = os.path.join(dirpath, item)

            if os.path.isdir(item_path):
                dirs.append(f"📁 {item}/")
            else:
                # Get file size
                size = os.path.getsize(item_path)
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/1024/1024:.1f}MB"

                files.append(f"📄 {item} ({size_str})")

        # Format output
        result = f"Contents of '{directory or ALLOWED_READ_DIR}':\n\n"

        if dirs:
            result += "Directories:\n"
            result += "\n".join(dirs) + "\n\n"

        if files:
            result += "Files:\n"
            result += "\n".join(files)

        print(f"[FILE TOOL] ✓ Listed {len(files)} files, {len(dirs)} directories")
        return result

    except PermissionError:
        return f"Error: Permission denied accessing directory"

    except Exception as e:
        return f"Error listing directory: {str(e)}"


# =============================================================================
# WRITE TOOLS
# =============================================================================

@tool
def write_file(filename: str, content: str) -> str:
    """
    Write content to a text file.

    Use this tool when the user asks to:
    - Create a file
    - Write to a file
    - Save output
    - Generate a report

    Examples:
    - "Write a summary report to summary.txt"
    - "Save these findings to results.json"
    - "Create a new config file"

    Args:
        filename: Name of file to create/overwrite
        content: Text content to write

    Returns:
        Success or error message

    For AI/ML Scientists:
    This allows the model to save its outputs to disk. Useful for generating
    reports, logs, or configuration files. In production, you might want
    human approval before writing files.
    """

    print(f"[FILE TOOL] Writing file: {filename}")

    # Ensure output directory exists
    ensure_directory_exists(ALLOWED_WRITE_DIR)

    # Build full path
    filepath = os.path.join(ALLOWED_WRITE_DIR, filename)

    # Security checks
    if not is_path_safe(filepath, ALLOWED_WRITE_DIR):
        return f"Error: Access denied. File must be in {ALLOWED_WRITE_DIR}"

    if not is_extension_allowed(filepath):
        return f"Error: File type not allowed. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"

    try:
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        file_size = len(content)
        print(f"[FILE TOOL] ✓ Wrote {file_size} characters")

        return f"Successfully wrote {file_size} characters to '{filename}' in {ALLOWED_WRITE_DIR}"

    except PermissionError:
        return f"Error: Permission denied writing to '{filename}'"

    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def append_to_file(filename: str, content: str) -> str:
    """
    Append content to an existing file (or create if doesn't exist).

    Use this tool for:
    - Adding to logs
    - Appending to reports
    - Cumulative outputs

    Args:
        filename: Name of file to append to
        content: Text to append

    Returns:
        Success or error message

    For AI/ML Scientists:
    Useful for streaming outputs or logging agent actions over time.
    Like appending to a log file during training.
    """

    print(f"[FILE TOOL] Appending to file: {filename}")

    ensure_directory_exists(ALLOWED_WRITE_DIR)

    filepath = os.path.join(ALLOWED_WRITE_DIR, filename)

    # Security checks
    if not is_path_safe(filepath, ALLOWED_WRITE_DIR):
        return f"Error: Access denied"

    if not is_extension_allowed(filepath):
        return f"Error: File type not allowed"

    try:
        # Append to file
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content)

        print(f"[FILE TOOL] ✓ Appended {len(content)} characters")

        return f"Successfully appended {len(content)} characters to '{filename}'"

    except Exception as e:
        return f"Error appending to file: {str(e)}"


# =============================================================================
# ADVANCED: FILE SEARCH TOOL
# =============================================================================

@tool
def search_files(query: str) -> str:
    """
    Search for files containing specific text.

    Use this when the user asks to:
    - Find files containing certain text
    - Search logs for errors
    - Locate files with specific content

    Args:
        query: Text to search for

    Returns:
        List of files containing the query

    For AI/ML Scientists:
    This is like grep for your agent. Searches all files in the allowed
    directory for matching text. Useful for log analysis or finding
    specific data files.
    """

    print(f"[FILE TOOL] Searching for: {query}")

    try:
        matches = []

        # Walk through all files in allowed directory
        for root, dirs, files in os.walk(ALLOWED_READ_DIR):
            for filename in files:
                filepath = os.path.join(root, filename)

                # Skip files that are too large
                if os.path.getsize(filepath) > MAX_FILE_SIZE_BYTES:
                    continue

                # Skip non-text files
                if not is_extension_allowed(filepath):
                    continue

                try:
                    # Search file contents
                    with open(filepath, 'r', encoding='utf-8') as f:
                        contents = f.read()

                        if query.lower() in contents.lower():
                            # Count occurrences
                            count = contents.lower().count(query.lower())
                            rel_path = os.path.relpath(filepath, ALLOWED_READ_DIR)
                            matches.append(f"{rel_path} ({count} occurrence(s))")

                except (UnicodeDecodeError, PermissionError):
                    # Skip files we can't read
                    continue

        if matches:
            result = f"Found '{query}' in {len(matches)} file(s):\n\n"
            result += "\n".join(matches)
            return result
        else:
            return f"No files found containing '{query}'"

    except Exception as e:
        return f"Error searching files: {str(e)}"


# =============================================================================
# SETUP & TESTING
# =============================================================================

def create_sample_files():
    """Creates sample files for testing."""

    print(f"[FILE TOOL] Creating sample files in {ALLOWED_READ_DIR}")

    ensure_directory_exists(ALLOWED_READ_DIR)

    # Sample file 1: Config
    config_content = """{
    "model": "mistral-7b",
    "endpoint": "sagemaker",
    "temperature": 0.7,
    "max_tokens": 500
}"""

    with open(os.path.join(ALLOWED_READ_DIR, "config.json"), 'w') as f:
        f.write(config_content)

    # Sample file 2: Log
    log_content = """2024-01-15 10:00:00 - INFO - Server started
2024-01-15 10:01:23 - INFO - Request processed successfully
2024-01-15 10:02:45 - ERROR - Connection timeout
2024-01-15 10:03:12 - INFO - Retrying connection
2024-01-15 10:03:30 - INFO - Connection established"""

    with open(os.path.join(ALLOWED_READ_DIR, "server.log"), 'w') as f:
        f.write(log_content)

    # Sample file 3: Report
    report_content = """# Monthly Sales Report

## Summary
- Total Revenue: $125,430
- Units Sold: 1,247
- Top Product: Widget A

## Details
Sales increased 15% compared to last month.
Strong performance in Widget A category."""

    with open(os.path.join(ALLOWED_READ_DIR, "report.md"), 'w') as f:
        f.write(report_content)

    print(f"[FILE TOOL] ✓ Created 3 sample files")


if __name__ == "__main__":
    """
    Test the file system tools.

    Usage:
        python file_system_tool.py
    """

    print("="*70)
    print("File System Tool Test")
    print("="*70)

    # Create sample files
    create_sample_files()

    # Test list files
    print("\n" + "="*70)
    print("Test 1: List Files")
    print("="*70)
    result = list_files.invoke("")
    print(result)

    # Test read file
    print("\n" + "="*70)
    print("Test 2: Read File")
    print("="*70)
    result = read_file.invoke("config.json")
    print(result)

    # Test write file
    print("\n" + "="*70)
    print("Test 3: Write File")
    print("="*70)
    result = write_file.invoke(
        "test_output.txt",
        "This is a test file created by the agent.\nIt contains multiple lines."
    )
    print(result)

    # Test search
    print("\n" + "="*70)
    print("Test 4: Search Files")
    print("="*70)
    result = search_files.invoke("ERROR")
    print(result)

    # Test append
    print("\n" + "="*70)
    print("Test 5: Append to File")
    print("="*70)
    result = append_to_file.invoke("test_output.txt", "\nAppended line.")
    print(result)

    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)
