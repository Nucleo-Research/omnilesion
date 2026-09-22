#!/usr/bin/env bash
# Grep the tree for things that must not be published: internal hosts and paths, credentials, e-mail addresses.
cd "$(dirname "$0")/.."
pattern='10\.0\.0\.|/workspace/results|/workspace/pancancer|virginia|nssh|luca-dev|\.ssh/|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH) PRIVATE|password|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}|codabench'
if grep -rInE --exclude-dir=.git --exclude=check_release_tree.sh "$pattern" . ; then
  echo "found strings that should not be published" >&2; exit 1
fi
echo "release tree clean"
