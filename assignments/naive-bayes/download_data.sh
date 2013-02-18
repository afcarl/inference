#!/usr/bin/env bash
# sets up spam, ham, and testing data in current directory

set -e

mkdir -p data/testing
oldpwd=$(pwd)
tmp=$(mktemp -d)

cd $tmp

# download ham/spam training data
curl -o enron5.tar.gz http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron5.tar.gz
tar xzvf enron5.tar.gz

mv enron5/{ham,spam} $oldpwd/data/

# download testing data from separate collection
curl -o enron1.tar.gz http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz
tar xzvf enron1.tar.gz

mv enron1/ham/* $oldpwd/data/testing/
mv enron1/spam/* $oldpwd/data/testing/

cd $oldpwd
rm -r $tmp

