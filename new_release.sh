#!/bin/bash

if [[ -z "$1" ]]; then
	echo "Set version tag"
	exit 1
fi

VERSION=$1

if [[ $VERSION == v* ]];then
	export VERSION=${VERSION#"v"}
fi

sed -i "s/\(^.*version=.*$\)/\ \ \ \ \ \ version='$VERSION',/" setup.py

git commit setup.py -m "New release"
git tag "v$VERSION"
git push
git push origin "v$VERSION"
python setup.py sdist
twine --upload dist/gfort2py-$VERSION.tar.gz

