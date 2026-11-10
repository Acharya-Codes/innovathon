
#!/bin/bash

# Python environment (Windows Git Bash)
PYTHON="./.venv/Scripts/python.exe"

# For Linux/WSL, use this instead:
# PYTHON="./.venv/bin/python"

OUTPUT_DIR="batch_outputs"

mkdir -p "$OUTPUT_DIR"

success=0
failed=0

echo "======================================"
echo "   BATCH RESUME EXTRACTION STARTED"
echo "======================================"

for i in {1..5}; do

    # Find resume1.pdf, resume1.png, etc.
    matches=(resume${i}.*)

    if [ ! -f "${matches[0]}" ]; then
        echo "[SKIPPED] resume${i} not found."
        failed=$((failed + 1))
        continue
    fi

    file="${matches[0]}"
    output="${OUTPUT_DIR}/resume${i}.json"

    echo ""
    echo "--------------------------------------"
    echo "Processing: $file"
    echo "--------------------------------------"

    if "$PYTHON" ingestor.py "$file" --output "$output"; then
        echo "[SUCCESS] Saved: $output"
        success=$((success + 1))
    else
        echo "[FAILED] Could not process: $file"
        failed=$((failed + 1))
    fi

done

echo ""
echo "======================================"
echo "         BATCH PROCESSING DONE"
echo "======================================"
echo "Successful: $success"
echo "Failed/Skipped: $failed"
echo "Output folder: $OUTPUT_DIR"
echo "======================================"
