 
# Lean how to use git commands

## Local git commands
### Add and commit file to local git

**add all modified files to local git**
git add .a

**commit all files with commit message**
git commit -m "Commit message"

### Distant git commands

** How to link local project to gitlab repository**

*1. generate ssh key*
ssh-keygen -t ed25519 -C "<put comment>"
/home/<user_name>/.ssh/gitlab_ed25519 # store public and private key on .ssh file
>do not need passphrase

*2. copy/paste public key on gitlab project*
    connect to gitlab account
    click on avatar icon on top right corner
    click on "Préférences"
    left headband, click on "Access" -> "SSH key"
    nn local terminal, write
vi /home/<user_name>/.ssh/gitlab_ed25519.pub
    copy paste : ssh-ed25519 <ssh_key>
    on gitlab, 
    Add new key and remove deadtime limit

*3. Start ssh agent and add keygen*
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
