///! Note some of the following structured are copied from settings-schema-derive. They coudn't get
///! reused because "`proc-macro` crate types currently cannot export any items other than functions
///! tagged with `#[proc_macro]`, `#[proc_macro_derive]`, or `#[proc_macro_attribute]`"
// This is needed by the code generated by the SettingsSchema macro
pub use serde::{Deserialize, Serialize};
pub use serde_json::to_value as to_json_value;
pub use settings_schema_derive::SettingsSchema;

/// The `Switch` is used to represent something that makes sense to specify its state only when it's enabled.
/// This should be used differently than `Option(al)`, that represent a value that can be omitted.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "state", content = "content")]
pub enum Switch<T> {
    Enabled(T),
    Disabled,
}

impl<T> Switch<T> {
    pub fn into_option(self) -> Option<T> {
        match self {
            Self::Enabled(t) => Some(t),
            Self::Disabled => None,
        }
    }
}

/// Type used to specify the default value for type `Option`.  
/// It allows specifying the set state and its content when it is set.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OptionalDefault<C> {
    pub set: bool,
    pub content: C,
}

/// Type used to specify the default value for type `Switch`.  
/// It allows setting the enabled state and its content when set to enabled.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SwitchDefault<C> {
    pub enabled: bool,
    pub content: C,
}

/// Type used to specify the default value for type `Vec`.  
/// It allows setting the default for the vector (all elements) and the default value for new elements.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VectorDefault<T> {
    pub element: T,
    pub content: Vec<T>,
}

/// Type used to specify the default value for type `Vec<(String, X)>`.  
/// It allows setting the default for the dictionary (all entries) and the default key and value for new entries.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DictionaryDefault<T> {
    pub key: String,
    pub value: T,
    pub content: Vec<(String, T)>,
}

/// GUI type associated to a numeric node.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NumericGuiType {
    TextBox,
    UpDown,
    Slider,
}

/// GUI type associated to a choice higher-order setting. Choice nodes cannot support this due to a
/// limitation of derive macros
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ChoiceControlType {
    Dropdown,
    ButtonGroup,
}

/// Data associated to a named or unnamed field. Can be set to advanced through the attribute `#[schema(advanced)]`
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EntryData {
    pub advanced: bool,
    pub content: SchemaNode,
}

// Description of the higher-order setting data. Choice, Bool and Action are all coerced to a
// numeric value when used to assign a value: Choice -> index of variant, bool -> [0,1], Action -> 0
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", content = "content")]
pub enum HigherOrderType {
    Choice {
        default: String,
        variants: Vec<String>,
        gui: Option<ChoiceControlType>,
    },
    Boolean {
        default: bool,
    },
    Action,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", content = "content")]
pub enum EntryType {
    Data(EntryData),
    HigherOrder {
        data_type: HigherOrderType,

        // Pseudocode interpreted by the UI
        modifiers: Vec<String>,
    },
    Placeholder,
}

/// Schema base type returned by `<YourStructOrEnum>::schema()`, generated by the macro
/// `#[derive(SettingsSchema)]`. It can be used as is (for Rust based GUIs) or it can be serialized
/// to JSON for creating GUIs in other languages.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", content = "content")]
pub enum SchemaNode {
    Section(Vec<(String, EntryType)>),

    Choice {
        default: String,
        variants: Vec<(String, Option<EntryData>)>,
        gui: Option<ChoiceControlType>,
    },

    Optional {
        default_set: bool,
        content: Box<SchemaNode>,
    },

    Switch {
        default_enabled: bool,
        content_advanced: bool,
        content: Box<SchemaNode>,
    },

    Boolean {
        default: bool,
    },

    Integer {
        default: i128,
        min: Option<i128>,
        max: Option<i128>,
        step: Option<i128>,
        gui: Option<NumericGuiType>,
    },

    Float {
        default: f64,
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
        gui: Option<NumericGuiType>,
    },

    Text {
        default: String,
    },

    // Instead of { schemaElement, length } a Vec is used. This is because each element can have a
    // different settings default.
    Array(Vec<SchemaNode>),

    Vector {
        default_element: Box<SchemaNode>,

        // This contains the settings default representation. It is untyped because the actual type
        // will be generated at compile time
        default: serde_json::Value,
    },

    Dictionary {
        default_key: String,
        default_value: Box<SchemaNode>,

        // This contains the settings default representation. It is untyped because the actual type
        // will be generated at compile time
        default: serde_json::Value,
    },
}
