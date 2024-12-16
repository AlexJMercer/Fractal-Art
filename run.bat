rmdir /s /q build

mkdir build

cmake -S . -B build -G "Visual Studio 17 2022"

cmake --build build --config Release

build\Release\Fractals.exe