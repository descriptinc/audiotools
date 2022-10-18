# remove doc/_build/html if present
rm -rf _build/html

#create a new directory (in doc/)
mkdir -p _build/html

# clone the entire repo into this directory (yes, this duplicates it)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/descriptinc/lyrebird-audiotools.git _build/html

# set this directory to track gh-pages
cd _build/html
git symbolic-ref HEAD refs/heads/gh-pages
rm .git/index
git clean -fdx
cd ../..

# in docs/, run `make html` to generate our doc, which will fill
# _build/html, but not overwrite the .git directory
make html

# now, add these bad-boys to the gh-pages repo, along with .nojekyll:
cd _build/html
git add .
git commit -m 'Updating docs'
git push origin gh-pages --force
