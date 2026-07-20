//! Reading documents out of whatever the user downloaded.
//!
//! Three formats, because those are the three the Hub and the classic corpora
//! actually ship in: parquet (FineWeb-Edu, DCLM, the Pile mirrors), JSON Lines,
//! and plain text. Everything downstream sees one flat stream of documents, so
//! adding a fourth is one match arm.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use parquet::record::reader::RowIter;
use parquet::schema::types::Type;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Parquet(parquet::errors::ParquetError),
    /// The file has no column or JSON key by the requested name.
    NoField {
        path: PathBuf,
        field: String,
    },
    /// A path whose extension is none of parquet/jsonl/json/txt.
    Unknown(PathBuf),
}

/// A set of document files, walked in sorted order so a run is reproducible.
#[derive(Debug)]
pub struct Corpus {
    paths: Vec<PathBuf>,
    field: String,
}

impl Corpus {
    /// Collect every readable file under `roots`, recursively.
    ///
    /// `field` names the parquet column or JSON key holding the document text —
    /// `text` for FineWeb-Edu and most of the Hub, `content` for some code sets.
    pub fn open(roots: &[PathBuf], field: &str) -> Result<Self, Error> {
        let mut paths = Vec::new();
        for root in roots {
            collect(root, &mut paths)?;
        }
        paths.sort();
        Ok(Self { paths, field: field.to_owned() })
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Every document in every file, in path order.
    pub fn docs(&self) -> impl Iterator<Item = Result<String, Error>> + '_ {
        self.paths.iter().flat_map(|path| match reader(path, &self.field) {
            Ok(docs) => docs,
            // A file that cannot be opened yields exactly one error and then
            // ends, so one corrupt shard does not abort a week of tokenising.
            Err(error) => Box::new(std::iter::once(Err(error))),
        })
    }
}

type Docs = Box<dyn Iterator<Item = Result<String, Error>>>;

fn reader(path: &Path, field: &str) -> Result<Docs, Error> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("parquet") => parquet_docs(path, field),
        Some("jsonl" | "json") => jsonl_docs(path, field),
        Some("txt" | "text" | "md") => {
            Ok(Box::new(std::iter::once(std::fs::read_to_string(path).map_err(Error::Io))))
        }
        _ => Err(Error::Unknown(path.to_owned())),
    }
}

fn parquet_docs(path: &Path, field: &str) -> Result<Docs, Error> {
    let reader = SerializedFileReader::new(File::open(path).map_err(Error::Io)?)?;
    let schema = reader.metadata().file_metadata().schema();
    let column = schema
        .get_fields()
        .iter()
        .find(|f| f.name() == field)
        .ok_or_else(|| Error::NoField { path: path.to_owned(), field: field.to_owned() })?
        .clone();

    // Projecting to the one column matters: a FineWeb-Edu row carries a dozen
    // metadata fields, and decoding them would dominate the tokenising pass.
    let projection = Type::group_type_builder("schema").with_fields(vec![column]).build()?;
    let rows = RowIter::from_file_into(Box::new(reader)).project(Some(projection))?;

    Ok(Box::new(rows.map(|row| Ok(row?.get_string(0)?.clone()))))
}

fn jsonl_docs(path: &Path, field: &str) -> Result<Docs, Error> {
    let file = BufReader::new(File::open(path).map_err(Error::Io)?);
    let (path, field) = (path.to_owned(), field.to_owned());
    Ok(Box::new(file.lines().filter(|line| !matches!(line, Ok(l) if l.trim().is_empty())).map(
        move |line| {
            let line = line.map_err(Error::Io)?;
            let value: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| Error::Io(io::Error::new(io::ErrorKind::InvalidData, e)))?;
            value[&field]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| Error::NoField { path: path.clone(), field: field.clone() })
        },
    )))
}

fn collect(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), Error> {
    if root.is_file() {
        out.push(root.to_owned());
        return Ok(());
    }
    for entry in std::fs::read_dir(root).map_err(Error::Io)? {
        collect(&entry.map_err(Error::Io)?.path(), out)?;
    }
    Ok(())
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<parquet::errors::ParquetError> for Error {
    fn from(error: parquet::errors::ParquetError) -> Self {
        Self::Parquet(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Parquet(error) => write!(f, "{error}"),
            Self::NoField { path, field } => {
                write!(f, "{} has no field `{field}`", path.display())
            }
            Self::Unknown(path) => write!(f, "{}: unknown format", path.display()),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            std::fs::write(dir.path().join(name), body).unwrap();
        }
        dir
    }

    #[test]
    fn jsonl_yields_one_document_per_line() {
        let dir = dir(&[("a.jsonl", "{\"text\":\"one\"}\n{\"text\":\"two\"}\n")]);

        let corpus = Corpus::open(&[dir.path().to_owned()], "text").unwrap();

        let docs: Vec<_> = corpus.docs().map(Result::unwrap).collect();
        assert_eq!(docs, ["one", "two"]);
    }

    #[test]
    fn files_are_read_in_path_order() {
        let dir = dir(&[("b.txt", "second"), ("a.txt", "first")]);

        let corpus = Corpus::open(&[dir.path().to_owned()], "text").unwrap();

        let docs: Vec<_> = corpus.docs().map(Result::unwrap).collect();
        assert_eq!(docs, ["first", "second"]);
    }

    #[test]
    fn a_missing_field_names_the_file() {
        let dir = dir(&[("a.jsonl", "{\"body\":\"one\"}\n")]);
        let corpus = Corpus::open(&[dir.path().to_owned()], "text").unwrap();

        let error = corpus.docs().next().unwrap().unwrap_err();

        assert!(error.to_string().contains("a.jsonl"), "{error}");
    }
}
