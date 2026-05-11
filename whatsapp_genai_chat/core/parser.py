from pathlib import Path
import pandas as pd
from whatsappchattodf import WhatsappChatToDF


def parse_chat(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Chat file not found: {path}")
    chat = WhatsappChatToDF(str(path))
    df = chat.run()
    required = {"User", "Message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected parser output schema. Expected columns {required}, got: {list(df.columns)}")
    df = df.rename(columns={"User": "sender", "Message": "message"})
    df = df[df["sender"].notna() & df["message"].notna()].reset_index(drop=True)
    return df
