venv_name=".venv"

# Check if .venv folder exists
if [ ! -d "$venv_name" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$venv_name"
    source "$venv_name/bin/activate"
    
    # Install packages from requirements.txt
    pip install -r requirements.txt
else
    echo "Virtual environment already exists."
    source "$venv_name/bin/activate"
fi

python3 train_test_model.py --model 'efficient_net'