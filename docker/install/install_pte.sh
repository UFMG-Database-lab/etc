
# Eigen and GSL package
sudo apt install libeigen3-dev libgsl0-dev -y

git clone https://github.com/mnqu/PTE

rm PTE/data -r

# Replace the eigen3 local version to system version
sed -i -e 's/..\/eigen-[[:digit:]].[[:digit:]].[[:digit:]]/\/usr\/include\/eigen3/g' PTE/pte/makefile

#sudo ln -sf eigen3/Eigen Eigen
make -C PTE/pte/
make -C PTE/text2hin/
make -C PTE/text2vec/

path_pte=`pwd`

alias pte=/.${path_pte}/PTE/pte/pte
alias data2dl=/.${path_pte}/PTE/text2hin/data2dl
alias data2w=/.${path_pte}/PTE/text2hin/data2w
alias infer=/.${path_pte}/PTE/text2vec/infer