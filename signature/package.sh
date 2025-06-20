#!/bin/bash

# package.sh - Package ipt_tilde for macOS distribution
#
# - Must be run from the project root (where 'externals/', 'docs/', etc. are siblings)
# - 'externals/ipt~.mxo' must exist
#
# This script creates a DMG containing the relevant files

set -e

# Check that the script is run from the root folder
if [[ ! -d "externals" || ! -d "signature" ]]; then
  echo "This script must be run from the project root (typically 'ipt_tilde/') which contains 'externals/' and 'signature/'." >&2
  exit 1
fi

# Check that the mxo file has been built
if [[ ! -d "externals/ipt~.mxo" ]]; then
  echo "externals/ipt~.mxo does not exist! Make sure to build the project first." >&2
  exit 1
fi

BUILD_PARENT_DIR="build/maxlib_build"
BUILD_DIR="$BUILD_PARENT_DIR/ipt_tilde"
DMG_NAME="ipt_tilde.dmg"
DMG_PATH="dist/$DMG_NAME"

echo "Cleaning $BUILD_DIR ..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Copying relevant files"
cp -R docs "$BUILD_DIR/"
cp -R externals "$BUILD_DIR/"
cp -R help "$BUILD_DIR/"
cp -R media "$BUILD_DIR/"
cp LICENSE "$BUILD_DIR/"
cp README.md "$BUILD_DIR/"

echo "Creating package DMG at $DMG_PATH ..."
mkdir -p "dist"
rm -f "$DMG_PATH"
hdiutil create -fs HFS+ -srcfolder "$BUILD_PARENT_DIR" -volname "ipt_tilde" "$DMG_PATH"

echo "Package DMG created at $DMG_PATH. Done!"
