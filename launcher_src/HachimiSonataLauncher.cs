using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("Hachimi Sonata Launcher")]
[assembly: AssemblyProduct("Hachimi Sonata")]
[assembly: AssemblyDescription("Hachimi Sonata · 哈基米音律工坊 一键启动器")]
[assembly: AssemblyCompany("Hachimi Sonata")]
[assembly: AssemblyVersion("0.5.0.0")]
[assembly: AssemblyFileVersion("0.5.0.0")]

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        string script = Path.Combine(baseDir, "launcher.py");
        if (!File.Exists(script))
        {
            MessageBox.Show(
                "找不到 launcher.py。\r\n请把本程序放在项目根目录后重试。",
                "Hachimi Sonata",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return;
        }

        string winDir = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
        string[][] candidates = new string[][]
        {
            new string[] { Path.Combine(winDir, "pyw.exe"), "-3 \"" + script + "\"" },
            new string[] { "pythonw.exe", "\"" + script + "\"" },
            new string[] { Path.Combine(winDir, "py.exe"), "-3 \"" + script + "\"" },
            new string[] { "python.exe", "\"" + script + "\"" }
        };

        foreach (string[] candidate in candidates)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo(candidate[0], candidate[1]);
                psi.WorkingDirectory = baseDir;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                Process.Start(psi);
                return;
            }
            catch
            {
                // 尝试下一个 Python 启动方式
            }
        }

        MessageBox.Show(
            "未找到可用的 Python。\r\n请先安装 Python 3.10+，并确保 pyw.exe / pythonw.exe 可用。",
            "Hachimi Sonata",
            MessageBoxButtons.OK,
            MessageBoxIcon.Warning);
    }
}
