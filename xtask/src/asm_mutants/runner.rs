use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    process::Output,
};

use testdir::testdir;

use super::mutations::{self, Mutation};

/// Test harness for a specific architecture.
pub(super) trait Harness {
    /// Returns the source path for this architecture's assembly.
    fn source_path() -> &'static str;

    /// Generates all individual mutations from the given assembly.
    ///
    /// The mutations must satisfy the requirements documented on [`Mutation`].
    fn mutations(source: &str) -> Result<Vec<Mutation>, String>;

    /// Configures a `cargo test` to use the given mutated assembly (if any).
    fn with_mutation(cmd: &mut Command, mutated_asm: Option<&Path>);
}

pub(super) fn run<Arch: Harness>() -> Result<(), String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("xtask manifest has no parent")?
        .to_owned();
    let dir: PathBuf = testdir!();
    let target_dir = dir.join("target");

    let test_command = |mutated_asm: Option<&Path>, compile_only: bool| {
        let mut cmd = Command::new(&cargo);
        cmd.current_dir(&repo_root)
            .arg("test")
            .arg("--release")
            .arg("--lib")
            .env("CARGO_TARGET_DIR", &target_dir);

        if compile_only {
            cmd.arg("--no-run");
        }

        Arch::with_mutation(&mut cmd, mutated_asm);

        cmd
    };

    let source_path = repo_root.join(Arch::source_path());
    let source = fs::read_to_string(&source_path)
        .map_err(|error| format!("failed to read {}: {error}", source_path.display()))?;

    let mutations = Arch::mutations(&source)?;

    println!("Running baseline and {} assembly mutants", mutations.len());
    let baseline = test_command(None, false)
        .output()
        .map_err(|error| format!("failed to run baseline: {error}"))?;
    if !baseline.status.success() {
        return Err(format!(
            "unmutated baseline failed:\n{}",
            describe_failure(&baseline)
        ));
    }

    let mut killed = 0;
    let mut crashed = 0;
    let mut invalid = Vec::new();
    let mut survivors = Vec::new();
    for (index, mutation) in mutations.iter().enumerate() {
        let mutant_path = dir.join(format!("mutant-{index:04}.S"));
        fs::write(&mutant_path, mutation.apply(&source)?)
            .map_err(|error| format!("failed to write {}: {error}", mutant_path.display()))?;

        print!("[{}/{}] {} ... ", index + 1, mutations.len(), mutation);
        let compile = test_command(Some(&mutant_path), true)
            .output()
            .map_err(|error| format!("failed to compile {mutation}: {error}"))?;
        let test = if compile.status.success() {
            Some(
                test_command(Some(&mutant_path), false)
                    .output()
                    .map_err(|error| format!("failed to test {mutation}: {error}"))?,
            )
        } else {
            None
        };

        match classify(&compile, test.as_ref()) {
            TestResult::Killed => {
                killed += 1;
                println!("killed");
            }
            TestResult::Crashed => {
                crashed += 1;
                println!("crashed (killed)");
                if let Some(test) = &test {
                    eprintln!("{}", describe_failure(test));
                }
            }
            TestResult::Invalid => {
                invalid.push(mutation);
                println!("INVALID");
                eprintln!("{}", describe_failure(&compile));
            }
            TestResult::Survived => {
                println!("SURVIVED");
                survivors.push(mutation);
            }
        }
    }

    println!(
        "\nResult: {killed} killed, {crashed} crashed, {} invalid, {} survived",
        invalid.len(),
        survivors.len()
    );
    if invalid.is_empty() && survivors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "invalid mutants:\n{}\nsurviving mutants:\n{}",
            mutations::format(&invalid),
            mutations::format(&survivors),
        ))
    }
}

#[derive(Debug, Eq, PartialEq)]
enum TestResult {
    Killed,
    Crashed,
    Survived,
    Invalid,
}

fn classify(compile: &Output, test: Option<&Output>) -> TestResult {
    if !compile.status.success() {
        TestResult::Invalid
    } else {
        let test = test.expect("a compiled mutant must be tested");
        if test.status.success() {
            TestResult::Survived
        } else if test.status.code().is_none() {
            TestResult::Crashed
        } else {
            TestResult::Killed
        }
    }
}

fn describe_failure(output: &Output) -> String {
    let mut message = format!("exit status: {}", output.status);
    for (name, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        if !bytes.is_empty() {
            message.push_str(&format!("\n{name}:\n{}", String::from_utf8_lossy(bytes)));
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn output(success: bool) -> Output {
        let mut command = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            command.args(["/C", if success { "exit 0" } else { "exit 1" }]);
        } else {
            command.args(["-c", if success { "exit 0" } else { "exit 1" }]);
        }
        command.output().unwrap()
    }

    #[test]
    fn classifies_compile_and_test_results() {
        assert_eq!(classify(&output(false), None), TestResult::Invalid);
        assert_eq!(
            classify(&output(true), Some(&output(false))),
            TestResult::Killed
        );
        assert_eq!(
            classify(&output(true), Some(&output(true))),
            TestResult::Survived
        );
    }
}
