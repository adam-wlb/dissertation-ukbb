import java.io.*;
import java.util.ArrayList;

/**
 * @author Adam Barron
 * @author 160212899
 * Move all of the accelerometer files for the given subset to a seperate folder,
 * ready to be unzipped and have GGIR ran on them.
 */
public class CollectSubset {

    private static final File[] DATA_1 = new File("/data1/biobank/ukbb_1").listFiles();
    private static final File[] DATA_2 = new File("/data2/biobank/ukbb_2").listFiles();
    private static ArrayList<String> filePaths = new ArrayList<>();

    public static void main(String[] args) {
        if (args.length != 1)
            throw new IllegalArgumentException("Usage: <subset.csv>");
        String subset = args[0];
        ArrayList<String> ids = new ArrayList<>();

        System.out.println("Parsing csv file");
        try {
            BufferedReader br = new BufferedReader(new FileReader(subset));
            // Skip header row
            br.readLine();
            String currentLine;
            while ((currentLine = br.readLine()) != null) {
                String lineSplit[] = currentLine.split(",");
                if (lineSplit.length == 2)
                    ids.add(lineSplit[0]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        findAccelerometerFiles(ids);

        try {
            File copyScript = createCopyFilesScript(filePaths);
            runScript(copyScript);
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
        }
    }

    private static void findAccelerometerFiles(ArrayList<String> ids) {
        for (String s : ids) {
            System.out.println("Searching for " + s);
            if (search(DATA_1, s))
                break;
            search(DATA_2, s);
        }
    }

    private static boolean search(File[] files, String targetId) {
        for (File f : files) {
            if (f.isDirectory()) {
                search(f.listFiles(), targetId);
            } else {
                // Split on "_" to get ID component of file name
                String[] filename = f.getName().split("_");
                String id = filename[0];
                if (id.equals(targetId)) {
                    System.out.println("File found for ID " + targetId);
                    filePaths.add(f.getAbsolutePath());
                    return true;
                }
            }
        }
        return false;
    }

    private static File createCopyFilesScript(ArrayList<String> filePaths) throws IOException {
        System.out.println("Building script");
        File script = File.createTempFile("copyFiles", "sh");
        Writer streamWriter = new OutputStreamWriter(new FileOutputStream(script));
        PrintWriter printWriter = new PrintWriter(streamWriter);
        // Make output directory if not present
        printWriter.println("mkdir -p datadir");
        for (String s : filePaths) {
            printWriter.println("cp " + s + " datadir");
        }
        printWriter.close();
        return script;
    }

    private static void runScript(File script) throws IOException, InterruptedException {
        System.out.println("Building process");
        try {
            ProcessBuilder pb = new ProcessBuilder("bash", script.toString());
            pb.inheritIO();
            System.out.println("Copying files to /datadir...");
            Process p = pb.start();
            p.waitFor();
            System.out.println("Process completed");
        } finally {
            System.out.println("Deleting temporary file");
            script.delete();
        }
    }
}
