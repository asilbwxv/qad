import os

def dump_all_files(root_dir, output_file="all_files_dump.txt"):
    print("Scanning:", os.path.abspath(root_dir))
    print("Writing to:", os.path.abspath(output_file))

    with open(output_file, "w", encoding="utf-8") as out:
        for folder, dirs, files in os.walk(root_dir):

            # Skip hidden directories AND specific excluded directories
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d != "qad_env"
                and d != "__pycache__"
            ]

            for filename in files:

                # Skip hidden files and unwanted extensions
                if filename.startswith("."):
                    continue
                if filename.lower().endswith((".jpg", ".wav", ".jsonl")):
                    continue

                file_path = os.path.join(folder, filename)
                out.write(f"=== {file_path} ===\n")

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"[Could not read file: {e}]\n")

                out.write("\n\n")

    print("Done. Output saved to", os.path.abspath(output_file))

root = os.getcwd()
dump_all_files(root)

