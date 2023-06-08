CC = g++ -fcoroutines -fPIC -std=c++20 -ggdb
chordPlayer:
	cd ./lua-cjson/ && gcc fpconv.c -c -I ../lua/
	cd ./lua-cjson/ && gcc lua_cjson.c -c -I ../lua/
	cd ./lua-cjson/ && gcc strbuf.c -c -I ../lua/
	$(CC) chordPlayer.cpp -o chordPlayer \
	 ./midifile/lib/libmidifile.a \
	 -I ./midifile/include \
	 -I ./lua/ ./lua/src/liblua.a \
	 -I ./chcpy \
    ./lua-cjson/fpconv.o \
    ./lua-cjson/lua_cjson.o \
    ./lua-cjson/strbuf.o -ldl

