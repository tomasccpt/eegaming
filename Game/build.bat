taskkill /IM "eegaming.exe" /F
cl src\eegaming.c lib\raylibdll.lib /Iinc /Zi
start eegaming.exe
