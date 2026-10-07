#!/usr/bin/env bash
# Data root for SignRule-Decide (CLAUDE.md §3, §6; design-01 D5).
#
#   scripts/setup_data_root.sh init    # one-time: create layout, gocryptfs vault, symlinks
#   scripts/setup_data_root.sh mount   # after every reboot: mount the vault
#   scripts/setup_data_root.sh status
#
# Layout:
#   /data/signrule/vault            gocryptfs ciphertext
#   ~/.local/share/signrule/vault-plain   plaintext view (FUSE mount) holding raw/, interim/, gold/
#     (Ubuntu's AppArmor profile for fusermount3 only allows FUSE mounts under $HOME, /mnt,
#     /media, /run/user and /tmp, so the mount point lives under $HOME)
#   /data/signrule/data/private -> the plaintext view
#   /data/signrule/data/{raw,interim,gold} -> private/{raw,interim,gold}
#   /data/signrule/data/{processed,splits}  masked data, outside the vault
#   /data/signrule/runs             checkpoints and logs
#   <repo>/data -> /data/signrule/data, <repo>/runs -> /data/signrule/runs
#
# Passphrase: SIGNRULE_VAULT_PASSFILE (default ~/.config/signrule/vault.pass, mode 600). Without
# that file gocryptfs asks interactively. Nothing secret is printed.
set -euo pipefail

ROOT=/data/signrule
VAULT=$ROOT/vault
PLAIN=${SIGNRULE_VAULT_MOUNT:-$HOME/.local/share/signrule/vault-plain}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CONF=${XDG_CONFIG_HOME:-$HOME/.config}/signrule
PASSFILE=${SIGNRULE_VAULT_PASSFILE:-$CONF/vault.pass}
MARKER=.signrule-vault

is_mounted() { grep -qs " $PLAIN fuse.gocryptfs " /proc/self/mounts; }

pass_args() {
  if [[ -f "$PASSFILE" ]]; then echo "-passfile $PASSFILE"; fi
}

do_mount() {
  if is_mounted; then echo "vault already mounted at $PLAIN"; return; fi
  mkdir -p "$PLAIN"
  # shellcheck disable=SC2046
  gocryptfs -q $(pass_args) "$VAULT" "$PLAIN"
  if [[ "${1:-}" != "--init" && ! -f "$PLAIN/$MARKER" ]]; then
    echo "mounted, but marker missing: wrong vault?" >&2; exit 1
  fi
  echo "vault mounted at $PLAIN"
}

do_init() {
  [[ -d "$ROOT" && -w "$ROOT" ]] || { echo "need writable $ROOT (sudo mkdir -p $ROOT && sudo chown \$USER $ROOT)" >&2; exit 1; }
  mkdir -p "$VAULT" "$PLAIN" "$ROOT/data" "$ROOT/data/processed" "$ROOT/data/splits" "$ROOT/runs" "$CONF"
  chmod 700 "$CONF"
  if [[ ! -f "$VAULT/gocryptfs.conf" ]]; then
    if [[ ! -f "$PASSFILE" ]]; then
      umask 077
      head -c 48 /dev/urandom | base64 | tr -d '\n' > "$PASSFILE"
      echo "generated passphrase file $PASSFILE (move it to a password manager)"
    fi
    # The init output contains the master key: keep it out of the terminal.
    umask 077
    gocryptfs -init -q -passfile "$PASSFILE" "$VAULT" > "$CONF/vault-init.txt" 2>&1
    echo "vault initialised; recovery info in $CONF/vault-init.txt (mode 600)"
  fi
  do_mount --init
  mkdir -p "$PLAIN/raw" "$PLAIN/interim" "$PLAIN/gold"
  [[ -d "$ROOT/data/private" && ! -L "$ROOT/data/private" ]] && rmdir "$ROOT/data/private"
  [[ -L "$ROOT/data/private" ]] || ln -s "$PLAIN" "$ROOT/data/private"
  touch "$PLAIN/$MARKER"
  for d in raw interim gold; do
    [[ -L "$ROOT/data/$d" ]] || ln -s "private/$d" "$ROOT/data/$d"
  done
  for d in data runs; do
    if [[ -d "$REPO/$d" && ! -L "$REPO/$d" ]]; then
      if [[ -n "$(ls -A "$REPO/$d")" ]]; then
        echo "$REPO/$d is not empty; move its contents into $ROOT/$d first" >&2; exit 1
      fi
      rmdir "$REPO/$d"
    fi
    [[ -L "$REPO/$d" ]] || ln -s "$ROOT/$d" "$REPO/$d"
  done
  do_status
}

do_status() {
  if is_mounted; then echo "vault: mounted at $PLAIN"; else echo "vault: NOT mounted"; fi
  for d in data runs; do echo "repo/$d -> $(readlink "$REPO/$d" || echo '(not a symlink)')"; done
}

case "${1:-status}" in
  init) do_init ;;
  mount) do_mount ;;
  status) do_status ;;
  *) echo "usage: $0 init|mount|status" >&2; exit 2 ;;
esac
