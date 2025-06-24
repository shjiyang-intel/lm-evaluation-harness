# OpenVINO 
```
mkdir codes
cd codes
git clone git@github.com:shjiyang-intel/openvino.git
git clone git@github.com:shjiyang-intel/openvino.genai.git

# build openvino in codes folder
cmake -DOPENVINO_EXTRA_MODULES=./openvino.genai -DCPACK_ARCHIVE_COMPONENT_INSTALL=OFF -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON -S ./openvino -B ./build
cmake --build ./build --target package ie_wheel -j8 --config Release
cmake --install ./build --prefix ./install  --config Release

cd ./install
setupvars.bat
```

# OV tokenizer
```
cd openvino.genai/thirdparty/openvino_tokenizers
vim pyproject.toml
```
comment openvino dependency to 
```
dependencies = [
    # support of nightly openvino packages with dev suffix
#    "openvino~=2025.3.0.dev"
]
```

build tokenizer
```
python -m pip wheel . -w dist/
```
openvino_tokenizers wheel will be generated under `dist/`

# GenAI

```
cd openvino.genai

git submodule update --init --recursive

vim pyproject.toml
```

comment openvino dependency to 
```
[build-system]
requires = [
    "py-build-cmake==0.4.3",
#    "openvino~=2025.3.0.0.dev",
    "pybind11-stubgen==2.5.4",
    "cmake~=3.23.0; platform_system != 'Darwin' or platform_machine == 'x86_64'",
    "cmake~=3.24.0; platform_system == 'Darwin' and platform_machine == 'arm64'",
]
```
```
python -m pip wheel . -w dist/
```
genai wheel will be generated under `dist/`
