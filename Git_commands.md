 
# Learn how to use git commands

## Local git commands
### Add and commit file to local git

**add all modified files to local git**
```
git add .a
```

**add specific file(s) to local git**
```
git add <file_1> <file_2>
```

**commit all files with commit message**
```
git commit -m "Commit message"
```

### Distant git commands

**How to link local project to gitlab repository**
Do this four steps one time

*1. generate ssh key*
```
ssh-keygen -t ed25519 -C "<put_comment>"
```
```/home/<user_name>/.ssh/gitlab_ed25519``` to store public and private key on .ssh file
>do not need passphrase, click on "Enter" two times

*2. copy/paste public key on gitlab project*
- connect to gitlab account
- click on avatar icon on top right corner
- click on "Preferences"
- left headband, click on "Access" -> "SSH keys"
- on local terminal, write

```
vi /home/<user_name>/.ssh/gitlab_ed25519.pub
```
- copy paste : ```ssh-ed25519 <ssh_key>```
- on gitlab,
- add new key and remove deadtime limit

*3. Start ssh agent and add keygen*
```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/gitlab_ed25519
```

*4. Test connection to gitlab site*
```bash
ssh -T git@gitlab.com
<Welcome to GitLab, @user_name !>
```


### add distant git
```bash
git remote <name> <url>
```

### push to distant git
```bash
git push <remote_name> <distant_branch>
```

### pull from distant git
```bash
git pull <remote_name> <local_branch>
```
