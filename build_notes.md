
cmake -S ./ -B ./build               \
     -DCMAKE_INSTALL_PREFIX=./install  \
     -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
cmake --build build -t install