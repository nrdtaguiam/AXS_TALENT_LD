import os
import subprocess
from pathlib import Path

def setup_custom_model():
    """
    Generates an optimized Ollama Modelfile and creates the custom 'axis-llama' model.
    """
    utils_dir = Path(__file__).parent
    modelfile_path = utils_dir / "Modelfile"
    
    modelfile_content = (
        "FROM llama3:8b\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER num_ctx 8192\n"
        "PARAMETER num_batch 1024\n"
        "PARAMETER num_keep 2400\n"
        "PARAMETER num_predict 512\n"
    )
    
    # 1. Write the Modelfile configuration
    print(f"Writing Modelfile to: {modelfile_path}")
    try:
        modelfile_path.write_text(modelfile_content, encoding="utf-8")
        print("Successfully wrote Modelfile.")
    except Exception as e:
        print(f"Error writing Modelfile: {e}")
        return False
        
    # 2. Attempt to create the custom model using 'ollama' CLI
    print("Attempting to run 'ollama create axis-llama'...")
    try:
        # Run command: ollama create axis-llama -f <path_to_modelfile>
        result = subprocess.run(
            ["ollama", "create", "axis-llama", "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print("Successfully created custom local model 'axis-llama'.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing ollama command (exit code {e.returncode}):")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        print("Notice: Make sure Ollama is installed and running locally on your machine.")
        return False
    except FileNotFoundError:
        print("Notice: 'ollama' executable was not found on the system PATH.")
        print("Please install Ollama and run: ollama create axis-llama -f <Modelfile>")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during model creation: {e}")
        return False

if __name__ == "__main__":
    setup_custom_model()
