
#!/bin/bash

# Step 1: Create a conda environment with Python 3.11.10
conda create -n partialaams python=3.11.10 -y

# Step 2: Activate the conda environment
conda activate partialaams

# Step 3: Clone the GranMapache repository
git clone https://github.com/MarcosLaffitte/GranMapache.git

# Step 4: Change directory to GranMapache
cd GranMapache

# Step 5: Install GranMapache
pip install .

# Step 6: Return to the original directory
cd ..

# Step 7: Remove the GranMapache directory
rm -rf GranMapache

# Step 8: Install the requirements
pip install -r requirements.txt

echo "Installation complete. The 'partialaams' environment is ready to use."
