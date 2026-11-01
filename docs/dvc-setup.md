# DVC setup (Google Drive remote)

How a team member connects to the shared DVC storage.

Status: the remote is configured, but **no dataset is tracked yet**. The first
real use will be Dataset V1. Until then `dvc pull` has nothing to download.

## What is shared and what is secret

- Shared in Git: `.dvc/config` (the Drive folder id, which is not a secret).
- Secret, never in Git: the OAuth client id and client secret. They live in
  `.dvc/config.local`, which is git-ignored.
- The Drive folder is restricted. Only invited accounts can open it.

## One-time setup for a new member

1. Install the project: `pip install -e ".[dev,dvc]"`. The `dvc` extra pins the
   versions that make the Google Drive remote import correctly.
2. Ask the repo owner to:
   - share the Drive folder with your Google account (Viewer should be enough
     to pull, not tested with a teammate yet),
   - add your account as a test user of the Google Cloud OAuth app,
   - send you the client id and client secret through a private channel.
3. Store them locally. Replace the placeholders, never commit real values:

   ```bash
   dvc remote modify --local gdrive gdrive_client_id <CLIENT_ID>
   dvc remote modify --local gdrive gdrive_client_secret <CLIENT_SECRET>
   ```

4. Check that the secrets file is ignored:

   ```bash
   git check-ignore -v .dvc/config.local
   ```

5. Pull the data. The first run opens a browser to log in with the Google
   account that has access:

   ```bash
   dvc pull -j 1
   ```

   `-j 1` runs one job at a time, so you get a single login prompt.

## Good to know

- The default DVC Google app is blocked by Google, which is why the project has
  its own OAuth client.
- The OAuth app is in Testing mode. Logins may expire after a few days and
  need to be repeated.
- DVC asks for access to the whole Google Drive of the logged-in account, not
  only the shared folder. Prefer an account dedicated to this project.
- On Windows, DVC writes `.dvc/config` with CRLF line endings. The
  `mixed-line-ending` pre-commit hook fixes it and cancels the commit once. Run
  `git add .dvc/config` and commit again.
- If a client secret is ever committed or pasted in a public place, create a new
  one in Google Cloud and stop using the old one.
