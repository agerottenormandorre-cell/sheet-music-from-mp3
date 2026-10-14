"""Builds the desktop program: app.ico and the launcher "Ноты из записи.exe".

The launcher is a few lines of C# compiled by the C# compiler that ships with Windows
(.NET Framework 4); it starts gui.pyw with the .venv Python and passes on any files
dropped onto it. Run by setup.bat; can be re-run on its own.
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
ICON = ROOT / "app.ico"
EXE = ROOT / "Ноты из записи.exe"
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")

LAUNCHER = r'''
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;
using System.Reflection;

[assembly: AssemblyTitle("Ноты из записи")]
[assembly: AssemblyProduct("Ноты из записи")]
[assembly: AssemblyCompany("astramixofficial@gmail.com")]
[assembly: AssemblyCopyright("Автор: astramixofficial@gmail.com")]
[assembly: AssemblyVersion("1.0.0.0")]

class Launcher {
    [STAThread]
    static void Main(string[] args) {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string py = Path.Combine(dir, @".venv\Scripts\pythonw.exe");
        if (!File.Exists(py)) {
            MessageBox.Show("Не найдено окружение .venv. Запустите setup.bat.", "Ноты из записи",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }
        var cmd = new StringBuilder("\"" + Path.Combine(dir, "gui.pyw") + "\"");
        foreach (string a in args)  // a trailing backslash would escape the closing quote
            cmd.Append(" \"" + (a.EndsWith("\\") ? a + "\\" : a) + "\"");
        Process.Start(new ProcessStartInfo(py, cmd.ToString()) { UseShellExecute = false, WorkingDirectory = dir });
    }
}
'''


def lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def draw_icon(n=1024):
    """Violet tile, two beamed eighth notes over a keyboard."""
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    grad = Image.new("RGBA", (n, n))
    top, bottom = (124, 77, 255), (49, 27, 146)
    for y in range(n):
        grad.paste(lerp(top, bottom, y / n) + (255,), (0, y, n, y + 1))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle((32, 32, n - 32, n - 32), radius=220, fill=255)
    img.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(img)
    white = (255, 255, 255, 255)
    # keyboard
    kx0, kx1, ky0, ky1 = 170, n - 170, 600, 850
    d.rounded_rectangle((kx0, ky0, kx1, ky1), radius=40, fill=white)
    keys = 7
    w = (kx1 - kx0) / keys
    for i in range(1, keys):
        x = kx0 + i * w
        d.line((x, ky0 + 10, x, ky1 - 10), fill=(49, 27, 146, 255), width=10)
    for i in (1, 2, 4, 5, 6):
        x = kx0 + i * w
        d.rounded_rectangle((x - w * 0.3, ky0, x + w * 0.3, ky0 + 150), radius=14, fill=(40, 22, 120, 255))
    # beamed eighth notes
    head = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
    ImageDraw.Draw(head).ellipse((20, 50, 200, 170), fill=white)
    head = head.rotate(22, resample=Image.BICUBIC)
    for cx, cy in ((362, 470), (652, 420)):
        img.alpha_composite(head, (cx - 110, cy - 110))
    d.rectangle((425, 190, 455, 470), fill=white)
    d.rectangle((715, 140, 745, 420), fill=white)
    d.polygon([(425, 190), (745, 140), (745, 225), (425, 275)], fill=white)
    return img


def build_icon():
    big = draw_icon()
    sizes = [16, 24, 32, 48, 64, 128, 256]
    big.resize((256, 256), Image.LANCZOS).save(ICON, sizes=[(s, s) for s in sizes])
    big.resize((512, 512), Image.LANCZOS).save(ROOT / "app.png")


def build_exe():
    if not CSC.exists():
        sys.exit("C# compiler not found: " + str(CSC))
    src = ROOT / ".venv" / "launcher.cs"
    src.write_text(LAUNCHER, encoding="utf-8-sig")
    subprocess.run([str(CSC), "/nologo", "/target:winexe", "/optimize", "/codepage:65001",
                    f"/win32icon:{ICON}", f"/out:{EXE}", "/reference:System.Windows.Forms.dll", str(src)],
                   check=True)


def desktop_shortcut():
    ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut("
          "[Environment]::GetFolderPath('Desktop')+'\\Ноты из записи.lnk');"
          f"$s.TargetPath='{EXE}';$s.WorkingDirectory='{ROOT}';$s.IconLocation='{ICON}';$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


if __name__ == "__main__":
    build_icon()
    build_exe()
    if "--shortcut" in sys.argv:
        desktop_shortcut()
    print("OK:", EXE)
