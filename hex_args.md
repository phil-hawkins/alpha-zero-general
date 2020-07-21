Train on a 3x3 board from a blank slate

python ./hex_main.py --game_board_size 3 --arenaCompare 100 --load_folder './temp/gat' --load_file 'temp.pth.tar' --numMCTSSims 10 --updateThreshold 0.51 --numIters 5

Train on a 4x4 board

python ./hex_main.py --game_board_size 4 --arenaCompare 100 --load_model --load_folder './temp/gat' --load_file 'temp.pth.tar' --numMCTSSims 20 --updateThreshold 0.51 --numIters 5