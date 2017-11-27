#!/bin/bash

if [[ -z "$1" ]]; then
	echo "Set version tag"
	exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "master" ]]; then
  echo 'Not on master branch'
  exit 1
fi


VERSION=$1

if [[ $VERSION == v* ]];then
	export VERSION=${VERSION#"v"}
fi

sed -i "s/__version__\ =.*/__version__\ =\ '$VERSION'/" gfort2py/version.py
sed -i "s/Current\ stable\ version\ is\ .*/Current\ stable\ version\ is\ $VERSION/" README.md


git commit setup.py README.md -m "New release"
git tag v$VERSION
git push
git push origin v$VERSION
python setup.py sdist

echo "Now do: twine upload dist/gfort2py-$VERSION.tar.gz"

git checkout maint
git merge master
