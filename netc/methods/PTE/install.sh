sudo apt install libgsl-dev libgsl27 gsl-bin libeigen3-dev
git clone https://github.com/mnqu/PTE
sed -i -e 's/..\/eigen-[[:digit:]].[[:digit:]].[[:digit:]]/\/usr\/include\/eigen3\//g' PTE/pte/makefile
for d in pte text2hin text2vec;
do
    echo $d;
    cd PTE/$d; make; cd ../..
done
mkdir exec
cp PTE/pte/pte PTE/text2hin/data2w PTE/text2hin/data2dl PTE/text2vec/infer exec/
rm -rf PTE
