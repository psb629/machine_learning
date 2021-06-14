#!/bin/zsh

git_dir=~/Github
branch=main

work_dir=$git_dir/machine_learning
cd $work_dir
git init
git config --global user.name "psb629"
git config --global user.email "psb629@gmail.com"
git remote remove origin
git_id=psb629
## my_token, if you write this in one line, github will delete this token
aaa=69c34aad789ae9
bbb=72275d2ca4785
ccc=660d8e0d9aac8
git_password=${aaa}${bbb}${ccc} # personal access token
git remote add origin https://"$git_id":"$git_password"@github.com/psb629/machine_learning.git
git pull origin $branch

echo "> * `users`(`ipconfig getifaddr en0`): `date`" >>$work_dir/README.md
git add -A
git commit -m "ran setup.sh"
git push -u origin $branch
