# Script to download datasets from HuggingFace
# This script downloads datasets to a folder at the same directory level as KraftonLab repo
#
# Usage:
#   ./download.sh --name lafan1
#   ./download.sh --name lafan1 --dir /path/to/datasets
#   DATASET_DIR=/path/to/datasets ./download.sh --name lafan1
#
# Environment Variables:
#   DATASET_DIR: Base directory for datasets. Can be overridden with --dir flag.

# Default dataset
DATASET_NAME="lafan1"
CUSTOM_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dir)
            CUSTOM_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --name <dataset_name> [--dir <dataset_directory>]"
            echo ""
            echo "Options:"
            echo "  --name <name>  Dataset to download (default: lafan1)"
            echo "  --dir <path>   Directory to store datasets (default: script directory)"
            echo ""
            echo "Environment Variables:"
            echo "  DATASET_DIR    Base directory for datasets (overridden by --dir)"
            echo ""
            echo "Available datasets:"
            echo "  lafan1      - LAFAN1 Retargeting Dataset"
            echo "  amass_g1    - AMASS Retargeted for G1"
            echo "  phuma       - PHUMA Physically-Grounded Humanoid Locomotion Dataset"
            echo "  omnicontrol - OmniControl Generated Dataset"
            echo ""
            echo "Examples:"
            echo "  $0 --name lafan1"
            echo "  $0 --name lafan1 --dir /data/motion_datasets"
            echo "  DATASET_DIR=/data/motion_datasets $0 --name amass_g1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine dataset directory (priority: --dir flag > DATASET_DIR env > script directory)
if [ -n "$CUSTOM_DIR" ]; then
    DATASET_DIR="$CUSTOM_DIR"
    echo "Using directory from --dir flag: $DATASET_DIR"
elif [ -n "$DATASET_DIR" ]; then
    echo "Using directory from DATASET_DIR environment variable: $DATASET_DIR"
else
    DATASET_DIR="$SCRIPT_DIR"
    echo "Using default directory: $DATASET_DIR"
fi

# Create directory if it doesn't exist
mkdir -p "$DATASET_DIR"

echo "Dataset directory: $DATASET_DIR"

# Navigate to the dataset directory
cd "$DATASET_DIR" || exit 1

# Download the requested dataset
case $DATASET_NAME in
    lafan1)
        REPO_NAME="LAFAN1_Retargeting_Dataset"
        REPO_URL="https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"

        echo "Downloading LAFAN1 dataset from HuggingFace..."
        echo "Target directory: $(pwd)/$REPO_NAME"

        if [ -d "$REPO_NAME" ]; then
            echo "$REPO_NAME already exists. Skipping download."
            echo "If you want to re-download, please remove the directory first:"
            echo "  rm -rf $DATASET_DIR/$REPO_NAME"
        else
            git clone "$REPO_URL"

            if [ $? -eq 0 ]; then
                echo "Successfully downloaded LAFAN1 dataset!"
                echo "Dataset location: $DATASET_DIR/$REPO_NAME"
            else
                echo "Failed to download LAFAN1 dataset."
                exit 1
            fi
        fi
        ;;
    amass_g1)
        REPO_NAME="amass_g1"
        REPO_URL="https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1"

        echo "Downloading AMASS G1 dataset from HuggingFace..."
        echo "Target directory: $(pwd)/$REPO_NAME"

        if [ -d "$REPO_NAME" ]; then
            echo "$REPO_NAME already exists. Skipping download."
            echo "If you want to re-download, please remove the directory first:"
            echo "  rm -rf $DATASET_DIR/$REPO_NAME"
        else
            git clone "$REPO_URL" "$REPO_NAME"

            if [ $? -eq 0 ]; then
                echo "Successfully downloaded AMASS G1 dataset!"
                echo "Dataset location: $DATASET_DIR/$REPO_NAME"
            else
                echo "Failed to download AMASS G1 dataset."
                exit 1
            fi
        fi
        ;;
    phuma)
        REPO_NAME="PHUMA"
        REPO_URL="https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA"

        echo "Downloading PHUMA dataset from HuggingFace..."
        echo "Target directory: $(pwd)/$REPO_NAME"

        if [ -d "$REPO_NAME" ]; then
            echo "$REPO_NAME already exists. Skipping download."
            echo "If you want to re-download, please remove the directory first:"
            echo "  rm -rf $DATASET_DIR/$REPO_NAME"
        else
            git clone "$REPO_URL"

            if [ $? -eq 0 ]; then
                echo "Successfully downloaded PHUMA dataset!"
                echo "Dataset location: $DATASET_DIR/$REPO_NAME"

                # Unzip the data.zip file if it exists
                if [ -f "$REPO_NAME/data.zip" ]; then
                    echo "Extracting data.zip..."
                    cd "$REPO_NAME"
                    unzip -q data.zip
                    if [ $? -eq 0 ]; then
                        echo "Successfully extracted data.zip!"
                        echo "Data location: $DATASET_DIR/$REPO_NAME/data"
                    else
                        echo "Failed to extract data.zip."
                        cd "$DATASET_DIR"
                        exit 1
                    fi
                    cd "$DATASET_DIR"
                else
                    echo "Warning: data.zip not found in $REPO_NAME"
                fi
            else
                echo "Failed to download PHUMA dataset."
                exit 1
            fi
        fi
        ;;
    omnicontrol)
        REPO_NAME="OmniControl"
        REPO_URL="https://huggingface.co/datasets/KRAFTON/physical-ai-interactive-motion"
        TMP_DIR="physical-ai-interactive-motion"

        echo "Downloading OmniControl asset and dataset from HuggingFace..."
        echo "Target directory: $(pwd)/$REPO_NAME"

        if [ ! -d "$REPO_NAME" ]; then
            echo "ERROR: $REPO_NAME does not exist."
            exit 1
        fi

        # Remove TMP_DIR if it exists
        if [ -d "$TMP_DIR" ]; then
            echo "Found existing $TMP_DIR. Removing it first."
            rm -rf "$TMP_DIR"
        fi

        echo "[debug] pwd=$(pwd)"
        echo "[debug] cloning into $TMP_DIR"

        git clone "$REPO_URL" "$TMP_DIR" || { echo "git clone failed"; exit 1; }
        cd "$TMP_DIR" || { echo "cd failed"; exit 1; }

        if command -v git >/dev/null 2>&1; then
            echo "Pulling git-lfs raw files..."
            git lfs pull || { echo "git lfs pull failed"; exit 1; }
        else
            echo "ERROR: git is not installed."
            exit 1
        fi

        cd .. || exit 1

        echo "Copying contents into $REPO_NAME (no overwrite, no git files)..."

        rsync -av --ignore-existing \
            --exclude='.git/' \
            --exclude='.gitignore' \
            --exclude='.gitattributes' \
            "$TMP_DIR"/ "$REPO_NAME"/
        RSYNC_RC=$?

        echo "[debug] rsync exit code: $RSYNC_RC"

        # Clean up
        rm -rf "$TMP_DIR"

        if [ $RSYNC_RC -eq 0 ]; then
            echo "Successfully downloaded OmniControl dataset!"
            echo "Dataset location: $(pwd)/$REPO_NAME"
        else
            echo "Failed to copy OmniControl dataset (rsync failed)."
            exit 1
        fi
        ;;
    *)
        echo "Error: Unknown dataset name '$DATASET_NAME'"
        echo "Use --help to see available datasets"
        exit 1
        ;;
esac

echo "Done!"
