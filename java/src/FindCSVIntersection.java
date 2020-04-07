import java.io.*;
import java.util.*;

/**
 * @author Adam Barron
 * @author 160212899
 * Produces a new CSV file with class labels of those patients with accelerometer data.
 */
public class FindCSVIntersection {

    public static void main(String[] args) {
        if (args.length != 3)
            throw new IllegalArgumentException("Usage: <class_labels> <accelerometer_ids> <output_csv");

        String classLabelInput = args[0];
        String idsInput = args[1];
        String outFile = args[2];

        TreeMap<Integer, Integer> classLabels = null;
        ArrayList<Integer> ids = null;
        try {
            classLabels = sortClassLabels(classLabelInput);
            ids = sortIds(idsInput);
            outputIntersection(classLabels, ids, outFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static TreeMap<Integer, Integer> sortClassLabels(String filepath) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filepath));
        // TreeMaps automatically sort keys
        TreeMap<Integer, Integer> classLabels = new TreeMap<>();

        // Skip header row
        String headerLine = br.readLine();
        String currentLine;
        while ((currentLine = br.readLine()) != null) {
            String lineSplit[] = currentLine.split(",");
            if (lineSplit.length == 2) {
                int[] ints = new int[2];
                ints[0] = Integer.valueOf(lineSplit[0]);
                ints[1] = Integer.valueOf(lineSplit[1]);
                classLabels.put(ints[0], ints[1]);
            }
        }
        br.close();
        return classLabels;
    }

    private static ArrayList<Integer> sortIds(String filepath) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filepath));
        ArrayList<Integer> ids = new ArrayList<>();

        String currentLine;
        while ((currentLine = br.readLine()) != null) {
            // Ignore non integers
            try {
                Integer i = Integer.valueOf(currentLine);
                ids.add(i);
            } catch (NumberFormatException e) {
                System.out.println("WARNING: Non-integer ID found: " + currentLine);
            }
        }
        br.close();
        Collections.sort(ids);
        return ids;
    }

    private static void outputIntersection(TreeMap<Integer, Integer> classLabels, ArrayList<Integer> ids, String outfile) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(outfile));

        // Add header row
        bw.write("eid,HEALTHY_CVD_T2D1_T2D2\n");
        for (Integer i : ids) {
            if (classLabels.containsKey(i)) {
                bw.write(i + "," + classLabels.get(i) + "\n");
            }
        }
        bw.close();
    }

}
