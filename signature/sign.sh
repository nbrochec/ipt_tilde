#!/bin/bash

show_help() {
  cat <<EOF
sign.sh - Code sign and notarize externals/ipt~.mxo for macOS

Usage:
  ./signature/sign.sh <codesign-identity> <notarytool-keychain-profile>

This script signs externals/ipt~.mxo, creates a DMG (ipt_tilde.dmg) in build/, submits it for notarization, and staples the result.

Arguments:
  <codesign-identity>           Name of the codesign identity to use
  <notarytool-keychain-profile> Name of the keychain profile for notarytool authentication

Requirements:
  - Must be run from the project root (where 'externals/' and 'signature/' are siblings)
  - 'externals/ipt~.mxo' must exist

Example:
  ./signature/sign.sh "Developer ID Application: Your Name (TEAMID)" "your-keychain-profile"
EOF
}

set -e

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
  exit 0
fi

# Color codes
GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m" # No Color

info() {
  echo -e "${GREEN}[INFO] $*${NC}"
}

error() {
  echo -e "${RED}[ERROR] $*${NC}" >&2
}

# Check for two arguments
if [[ $# -ne 2 ]]; then
  error "Two arguments required: <codesign-identity> <notarytool-keychain-profile>"
  error "Run with --help for usage information." >&2
  exit 1
fi

CODESIGN_IDENTITY="$1"
NOTARY_PROFILE="$2"

# Variables
MXO_PATH="externals/ipt~.mxo"
DMG_NAME="ipt_tilde.dmg"
DMG_PATH="build/$DMG_NAME"
ENTITLEMENTS="signature/entitlements.plist"



# Check that the script is run from the root folder
if [[ ! -d "externals" || ! -d "signature" ]]; then
  error "This script must be run from the project root (typically 'ipt_tilde/') which contains 'externals/' and 'signature/'."
  exit 1
fi

# Check that the mxo file has been built
if [[ ! -d "$MXO_PATH" ]]; then
  error "$MXO_PATH does not exist! Make sure to build the project first."
  exit 1
fi

info "Codesigning $MXO_PATH ..."
codesign --force --deep --timestamp -s "$CODESIGN_IDENTITY" --options=runtime --entitlements "$ENTITLEMENTS" "$MXO_PATH"

info "Creating DMG at $DMG_PATH ..."
mkdir -p build
hdiutil create "$DMG_PATH" -fs HFS+ -srcfolder "$MXO_PATH" -ov

info "Submitting DMG for notarization ..."
xcrun notarytool submit "$DMG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait

info "Stapling notarization ticket to $MXO_PATH ..."
xcrun stapler staple "$MXO_PATH"

info "Codesigning, notarization, and stapling complete!"
