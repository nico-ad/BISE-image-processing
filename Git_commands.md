 
# ----- Add and commit file to .git

# add all modified files to .git
git add .a

# commit all files with commit message
git commit -m "Commit message"

# ----- distant git

# link gitlab storage to local storage

# 1. generate ssh key
ssh-keygen -t ed25519 -C "<put comment>"
/home/abad-ale/.ssh/gitlab_ed25519 # store public and private key on .ssh file
# do not need passphrase

# 2. copy/paste public key on gitlab project
# -> connect to gitlab account
# -> click on avatar icon on top right corner
# -> click on "Préférences"
# -> Left bandeau, click on "Accés" -> "Clé SSH"
# -> On local terminal,
vi /home/abad-ale/.ssh/gitlab_ed25519.pub
# copy paste : ssh-ed25519 <ssh_key>
# -> Add new key and remove deadtime limit

# 3. Start ssh agent and add keygen
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/gitlab_ed25519

# 4. Test connection to gitlab site
ssh -T git@gitlab.com
# message : Welcome to GitLab, @user_name !

# add distant git
git remote <name> <url>

# push to distant git
git push <remote_name> <distant_branch>

# pull from distant git
git pull <remote_name> <local_branch>
