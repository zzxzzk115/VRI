-- Google's GameActivity 4.4.2 (Apache-2.0), from the official Google Maven artifact:
-- https://dl.google.com/dl/android/maven2/androidx/games/games-activity/4.4.2/games-activity-4.4.2.aar
-- No local patches. The Android examples link the AAR's Prefab static module (GameActivity +
-- native_app_glue); the APK pulls the matching Java classes from Google Maven
-- (examples/android/build.gradle pins the same version).
package("game-activity")
    set_homepage("https://developer.android.com/games/agdk/game-activity")
    set_description("Android GameActivity and native app glue, statically linked")
    set_license("Apache-2.0")
    add_urls("https://dl.google.com/dl/android/maven2/androidx/games/games-activity/$(version)/games-activity-$(version).aar")
    add_versions("4.4.2", "16069bb61a34e9cc7a4f6b4cef0ddb9e2f3f61fe781b282787f574f8880ba38e")
    add_links("game-activity_static")
    add_syslinks("android", "log")

    on_install("android", function (package)
        import("utils.archive")
        archive.extract(package:originfile(), "aar", {extension = ".zip"})
        os.cd("aar")
        local module = "prefab/modules/game-activity_static"
        os.cp(module .. "/include/*", package:installdir("include"))
        os.cp(module .. "/libs/android." .. package:arch() .. "/libgame-activity_static.a", package:installdir("lib"))
    end)
package_end()
