[app]
title = Adobe Becker Style Studio
package.name = adobebeckerstudio
package.domain = org.hcak
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3,kivy,numpy==v1.26.4
orientation = landscape
fullscreen = 0

android.permissions = INTERNET
android.api = 35
android.minapi = 24
android.build_tools_version = 35.0.0
android.archs = arm64-v8a, armeabi-v7a

[buildozer]
log_level = 2
