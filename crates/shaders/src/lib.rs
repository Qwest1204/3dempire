use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc::{self, Receiver},
    thread,
    time::Instant,
};
pub struct ShaderWatcher {
    _watcher: RecommendedWatcher,
    events: Receiver<notify::Result<notify::Event>>,
}

#[derive(Debug)]
pub enum SlangCompileResult {
    Unchanged,
    Compiling,
    Compiled,
    Failed(String),
}

#[derive(Clone, Debug, Default)]
pub struct SlangHotReloadStatus {
    pub compiling: bool,
    pub last_error: Option<String>,
    pub last_output: Option<String>,
    pub last_success: Option<Instant>,
}

#[derive(Clone, Copy, Debug)]
pub struct ShaderAssetInfo {
    pub name: &'static str,
    pub source: &'static str,
    pub entries: &'static str,
    pub definitions: &'static str,
    pub variants: &'static str,
}

pub const SHADER_ASSETS: &[ShaderAssetInfo] = &[
    ShaderAssetInfo {
        name: "Sky background",
        source: "shaders/slang/debug_output.slang",
        entries: "vs_main, fs_main",
        definitions: "ENGINE_DEFERRED=1",
        variants: "default",
    },
    ShaderAssetInfo {
        name: "Deferred renderer",
        source: "shaders/slang/deferred.slang (+ common.slang)",
        entries: "gbuffer_vs, gbuffer_fs, fullscreen_vs, composite_fs, debug_fs",
        definitions: "ENGINE_DEFERRED=1",
        variants: "final, debug",
    },
];

struct CompileOutcome {
    result: Result<(), String>,
    output: String,
}

pub struct SlangHotReload {
    watcher: ShaderWatcher,
    workspace_root: PathBuf,
    results: Receiver<CompileOutcome>,
    result_sender: mpsc::Sender<CompileOutcome>,
    status: SlangHotReloadStatus,
}

impl SlangHotReload {
    pub fn new(workspace_root: impl Into<PathBuf>) -> notify::Result<Self> {
        let workspace_root = workspace_root.into();
        let (result_sender, results) = mpsc::channel();
        Ok(Self {
            watcher: ShaderWatcher::new(workspace_root.join("shaders/slang"))?,
            workspace_root,
            results,
            result_sender,
            status: SlangHotReloadStatus::default(),
        })
    }
    pub fn update(&mut self) -> SlangCompileResult {
        if let Ok(result) = self.results.try_recv() {
            self.status.compiling = false;
            self.status.last_output = Some(result.output);
            return match result.result {
                Ok(()) => {
                    self.status.last_error = None;
                    self.status.last_success = Some(Instant::now());
                    SlangCompileResult::Compiled
                }
                Err(error) => {
                    self.status.last_error = Some(error.clone());
                    SlangCompileResult::Failed(error)
                }
            };
        }
        if self.watcher.has_changes() {
            self.request_compile();
        }
        if self.status.compiling {
            SlangCompileResult::Compiling
        } else {
            SlangCompileResult::Unchanged
        }
    }

    pub fn request_compile(&mut self) {
        if self.status.compiling {
            return;
        }
        self.status.compiling = true;
        let workspace_root = self.workspace_root.clone();
        let result_sender = self.result_sender.clone();
        thread::spawn(move || {
            let result = Command::new("cargo")
                .args(["run", "-p", "xtask", "--", "shaders"])
                .current_dir(workspace_root)
                .output()
                .map_err(|error| format!("could not invoke shader compiler task: {error}"));
            let outcome = match result {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                    let log = format!("{stdout}{stderr}");
                    if output.status.success() {
                        CompileOutcome {
                            result: Ok(()),
                            output: log,
                        }
                    } else {
                        CompileOutcome {
                            result: Err(log.clone()),
                            output: log,
                        }
                    }
                }
                Err(error) => CompileOutcome {
                    output: error.clone(),
                    result: Err(error),
                },
            };
            let _ = result_sender.send(outcome);
        });
    }

    pub fn status(&self) -> &SlangHotReloadStatus {
        &self.status
    }
}

impl ShaderWatcher {
    pub fn new(directory: impl AsRef<Path>) -> notify::Result<Self> {
        let (sender, events) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(sender)?;
        watcher.watch(directory.as_ref(), RecursiveMode::Recursive)?;
        Ok(Self {
            _watcher: watcher,
            events,
        })
    }

    pub fn has_changes(&self) -> bool {
        let mut changed = false;
        while self.events.try_recv().is_ok() {
            changed = true;
        }
        changed
    }
}
