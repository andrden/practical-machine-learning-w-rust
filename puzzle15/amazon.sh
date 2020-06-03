curl https://sh.rustup.rs -sSf | sh
git clone https://github.com/andrden/practical-machine-learning-w-rust.git
cd practical-machine-learning-w-rust/puzzle15
#sudo apt install libcairo2-dev
sudo apt install -y libpango1.0-dev libcairo2-dev libatk1.0-dev libgdk-pixbuf2.0-dev libgtk-3-dev
rustup override set nightly

env TORCH_CUDA_VERSION=10.1 cargo build --release
env TORCH_CUDA_VERSION=10.1 cargo run --release



