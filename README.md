# Chirpy Starter

[![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy)][gem]&nbsp;
[![GitHub license](https://img.shields.io/github/license/cotes2020/chirpy-starter.svg?color=blue)][mit]

When installing the [**Chirpy**][chirpy] theme through [RubyGems.org][gem], Jekyll can only read files in the folders
`_data`, `_layouts`, `_includes`, `_sass` and `assets`, as well as a small part of options of the `_config.yml` file
from the theme's gem. If you have ever installed this theme gem, you can use the command
`bundle info --path jekyll-theme-chirpy` to locate these files.

The Jekyll team claims that this is to leave the ball in the user’s court, but this also results in users not being
able to enjoy the out-of-the-box experience when using feature-rich themes.

To fully use all the features of **Chirpy**, you need to copy the other critical files from the theme's gem to your
Jekyll site. The following is a list of targets:

```shell
.
├── _config.yml
├── _plugins
├── _tabs
└── index.html
```

To save you time, and also in case you lose some files while copying, we extract those files/configurations of the
latest version of the **Chirpy** theme and the [CD][CD] workflow to here, so that you can start writing in minutes.

## Usage

Check out the [theme's docs](https://github.com/cotes2020/jekyll-theme-chirpy/wiki).

### Local development with VS Code

Open this repository with Docker Desktop running and use **Dev Containers: Reopen in Container**.
On Windows, enable Docker Desktop's WSL integration for Ubuntu if VS Code mounts its WSLg/Wayland socket.

The container's `onCreateCommand` makes `/usr/local/bundle` writable by the development user before
the image's built-in `postCreateCommand` installs the gems. After changing the container configuration,
run **Dev Containers: Rebuild and Reopen in Container**.

If an existing container reports `Bundler::PermissionError`, run the following in its terminal, then rebuild
to complete all setup commands:

```sh
sudo chown -R "$(id -u):$(id -g)" /usr/local/bundle
bundle install
```

Do not run `sudo bundle install`; it can leave newly installed gems owned by root.

Start the development server with the **Run Jekyll Server** task or:

```sh
bash tools/run.sh
```

The helper enables LiveReload and uses polling inside Docker, including cgroup v2 containers.
Polling is needed for edits on Windows bind mounts, which may not produce Linux filesystem events.
For a direct Jekyll command, use `bundle exec jekyll serve --livereload --force_polling`.
The `-o` flag only opens a browser; it does not enable change detection.

Open `http://localhost:4000` and ensure both **4000** (site) and **35729** (LiveReload) are forwarded
in VS Code's Ports view. The container configuration requests these exact local ports because
the injected LiveReload client connects to port 35729. Restart a server that was already running
without polling; configuration changes do not update an existing Jekyll process.

## Contributing

This repository is automatically updated with new releases from the theme repository. If you encounter any issues or want to contribute to its improvement, please visit the [theme repository][chirpy] to provide feedback.

## License

This work is published under [MIT][mit] License.

[gem]: https://rubygems.org/gems/jekyll-theme-chirpy
[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy/
[CD]: https://en.wikipedia.org/wiki/Continuous_deployment
[mit]: https://github.com/cotes2020/chirpy-starter/blob/master/LICENSE
