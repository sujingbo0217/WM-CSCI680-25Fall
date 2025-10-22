#!/bin/bash
cat clone-repo.txt | xargs -P 16 -n 1 git clone
