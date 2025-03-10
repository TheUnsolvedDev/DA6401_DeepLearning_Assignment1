python3 train.py -e 25 -o adam -lr 0.0005 -w_d 0.0005 -nhl 3 -sz 128 -a tanh -loss cross_entropy -w_i xavier_normal
python3 train.py -e 25 -o adam -lr 0.0005 -w_d 0.0005 -nhl 3 -sz 128 -a tanh -loss squared_error -w_i xavier_normal
python3 train.py -e 25 -o adam -lr 0.0005 -w_d 0.0005 -nhl 3 -sz 128 -a tanh -loss cross_entropy -w_i xavier_normal -b 64 -d mnist
python3 train.py -e 25 -o adam -lr 0.0005 -w_d 0.0005 -nhl 4 -sz 64 -a tanh -loss cross_entropy -w_i xavier_normal -b 128 -d mnist
python3 train.py -e 25 -o nadam -lr 0.0005 -w_d 0.0005 -nhl 3 -sz 64 -a tanh -loss cross_entropy -w_i xavier_normal -b 64 -d mnist