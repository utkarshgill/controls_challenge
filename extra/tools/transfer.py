#!/usr/bin/env python3
"""
SCP file transfer helper for pushing files to another Mac over Ethernet.

Usage:
    python transfer.py <host> <file_or_dir> [file_or_dir ...] [--dest ~/Desktop] [--user engelbart] [--key ~/.ssh/id_ed25519]

Examples:
    python transfer.py 192.168.2.2 models/best.pt
    python transfer.py 192.168.2.2 experiments/exp050_rich_ppo/ --dest ~/Desktop/stuff
    python transfer.py 192.168.2.2 *.pt data/ --user engelbart --dest ~/results
"""

import argparse, subprocess, sys


def scp(host, user, key, paths, dest):
    base = ['scp', '-r', '-o', 'StrictHostKeyChecking=no']
    if key:
        base += ['-i', key]

    target = f'{user}@{host}:{dest}'
    cmd = base + paths + [target]
    print(f"$ {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


def main():
    p = argparse.ArgumentParser(description='SCP files to another Mac')
    p.add_argument('host', help='IP or hostname of target Mac')
    p.add_argument('paths', nargs='+', help='Files or directories to send')
    p.add_argument('--dest', default='~/Desktop', help='Remote destination (default: ~/Desktop)')
    p.add_argument('--user', default='engelbart', help='SSH user on target (default: engelbart)')
    p.add_argument('--key', default=None, help='SSH key path (optional)')
    args = p.parse_args()
    scp(args.host, args.user, args.key, args.paths, args.dest)


if __name__ == '__main__':
    main()
