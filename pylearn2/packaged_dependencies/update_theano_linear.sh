#!/bin/bash
#updates the packaged copy of TheanoLinear
#must be run from the packaged_dependencies directory

git rm -rf theano_linear
git clone https://github.com/jaberg/TheanoLinear.git TheanoLinearTemp
pushd TheanoLinearTemp
git archive master | tar -x -C ../
popd
rm -rf TheanoLinearTemp
rm README
find theano_linear | xargs git add -u

