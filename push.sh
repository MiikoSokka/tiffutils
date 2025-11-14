#!/usr/bin/env bash
set -euo pipefail

# Usage: ./git_push_package.sh "commit message" [branch]
if [[ $# -lt 1 ]]; then
  echo "❌ Error: You must provide a commit message."
  echo "Usage: $0 \"your commit message\" [branch]"
  exit 1
fi

COMMIT_MESSAGE="$1"
BRANCH="${2:-main}"

# 0) Validate repo and remote
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "❌ Not inside a Git repository."; exit 1; }

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "❌ No 'origin' remote configured."; exit 1;
fi

echo "📝 Commit message: $COMMIT_MESSAGE"
echo "🌿 Branch: $BRANCH"
echo "🔗 Remote: $(git remote get-url origin)"

# 1) Stage changes
echo "📂 Staging changes..."
git add -A

# 2) Show what will be committed
if ! git diff --cached --quiet; then
  echo "🧾 Staged changes:"
  git diff --cached --name-status
else
  echo "ℹ️ No changes staged; nothing to commit."
fi

# 3) Preflight: block if any staged file ≥ 95MB (GitHub hard limit is 100MB/file)
large_staged=$(
  git diff --cached --name-only -z \
  | xargs -0 -I{} bash -lc '[ -f "{}" ] && wc -c <"{}" || echo 0' \
  | awk '{s+=$1} END{print ""}' >/dev/null 2>&1 # warm up subshell
  git diff --cached --name-only -z \
  | xargs -0 -I{} bash -lc 'if [ -f "{}" ]; then size=$(wc -c <"{}"); if [ "$size" -ge 95000000 ]; then printf "%s\n" "{}"; fi; fi'
)

if [[ -n "${large_staged:-}" ]]; then
  echo "❌ One or more staged files are ≥ 95MB (GitHub rejects ≥ 100MB per file):"
  echo "$large_staged" | sed 's/^/   - /'
  echo "👉 Either remove them with 'git rm --cached <file>' or use Git LFS:"
  echo "   brew install git-lfs && git lfs install && git lfs track \"<patterns>\""
  exit 1
fi

# 4) Commit if there are staged changes
if ! git diff --cached --quiet; then
  echo "✅ Committing..."
  git commit -m "$COMMIT_MESSAGE" || true
else
  echo "✅ Nothing to commit."
fi

# 5) Push (verbose so server-side messages are visible)
echo "🚀 Pushing to GitHub..."
set +e
GIT_CURL_VERBOSE=1 git push --verbose origin "$BRANCH"
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  cat <<'EOF'
❌ Push failed.

Common causes:
  • A file ≥ 100MB was committed (GitHub rejects single files that big).
  • Large files that should be in Git LFS aren't tracked.
  • History contains large blobs from past commits.

What to try next:
  1) Show largest blobs in history:
     git rev-list --objects --all | git cat-file --batch-check='%(objectsize) %(rest)' | sort -n | tail -n 20
  2) If needed, migrate to LFS:
     brew install git-lfs
     git lfs install
     git lfs track "*.bam" "*.fa" "*.fa.gz" "*.fastq" "*.fq" "*.bigWig" "*.bw" "*.tiff" "*.ome.tiff" "*.h5" "*.hdf5" "*.npy" "*.npz"
     git add .gitattributes && git commit -m "Track large files with LFS"
     git lfs migrate import --include="*.bam,*.fa,*.fa.gz,*.fastq,*.fq,*.bw,*.bigWig,*.tiff,*.ome.tiff,*.h5,*.hdf5,*.npy,*.npz"
     git push --force-with-lease origin '"$BRANCH"'
EOF
  exit $rc
fi

echo "🎉 Done!"

