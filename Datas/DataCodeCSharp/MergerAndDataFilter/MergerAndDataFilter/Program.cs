using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MergerAndDataFilter
{
    class Program
    {

        static void Main(string[] args)
        {
            Dictionary<string, int[]> dictOfCountry = new Dictionary<string, int[]>() {

            { "E0", new int[] { 0, 23 } },
            { "E1", new int[] { 0, 23 } }, { "E2", new int[] { 0, 23 } }, { "E3", new int[] { 0, 23 } }, { "EC", new int[] { 0, 11 } },

            { "SC0", new int[] { 0, 22 } }, { "SC1", new int[] { 0, 22 } }, { "SC2", new int[] { 0, 19 } }, { "SC3", new int[] { 0, 19 } },

            { "D1", new int[] { 0, 23 } }, { "D2", new int[] { 0, 23 } },

            { "I1", new int[] { 0, 23 } }, { "I2", new int[] { 0, 19 } },

            { "SP1", new int[] { 0, 23 } }, { "SP2", new int[] { 0, 20 } },

            { "F1", new int[] { 0, 23 } }, { "F2", new int[] { 0, 20 } },

            { "N1", new int[] { 0, 23 } },

            { "B1", new int[] { 0, 21 } },

            { "P1", new int[] { 0, 22 } },

            { "T1", new int[] { 0, 22 } },

            { "G1", new int[] { 0, 22 } },

            };

            foreach (var dictKV in dictOfCountry)
            {
                string country = dictKV.Key;

                string initialPath = @"C:\DeepSoccerCurrentWork\";
                string outputPath = @"C:\DeepSoccerCurrentWork\Output\";

                string path = initialPath + country + " (";

                string[] lines = File.ReadAllLines(initialPath + country+ ".csv");

                Dictionary<string, int> dic = new Dictionary<string, int>();
                int j = 0;

                int numberofline = 0;

                List<string[]> list = new List<string[]>();

                dic.Add("Div", j);
                j++;
                dic.Add("Season", j);
                j++;

                for (int i = dictKV.Value[0]; i < dictKV.Value[1]+1; i++)
                {
                    if (i == dictKV.Value[0])
                    {
                        lines = File.ReadAllLines(initialPath + country + ".csv");
                    }
                    else { lines = File.ReadAllLines(path + i + ").csv"); }

                    foreach (string s in lines.First().Split(','))
                    {
                        if (!dic.ContainsKey(s))
                        {
                            dic.Add(s, j);
                            j++;
                        }
                    }


                    int c = 0;
                    string[] header = lines.First().Split(',');

                    foreach (string line in lines)
                    {
                        if (c != 0)
                        {
                            

                            list.Add(new string[124]);
                            list[numberofline][dic["Season"]] = (2017 - i).ToString();

                            int stringcount = 0;
                            foreach (string s in line.Split(','))
                            {
                                if (stringcount < header.Count())
                                {
                                    list[numberofline][dic[header[stringcount]]] = s;
                                }
                                stringcount++;
                            }
                            numberofline++;
                        }
                        c++;
                    }
                    //Console.WriteLine(i);
                }

                Console.WriteLine(list.Count);

                List<string[]> listWithoutEmpty = new List<string[]>();

                foreach (string[] li in list)
                {
                    if (!string.IsNullOrEmpty(li[dic["Date"]]) && !string.IsNullOrEmpty(li[dic["HomeTeam"]])
                           && !string.IsNullOrEmpty(li[dic["AwayTeam"]]) && !string.IsNullOrEmpty(li[dic["FTAG"]])
                           && !string.IsNullOrEmpty(li[dic["FTR"]]))
                    {
                        int annee = int.Parse(li[dic["Date"]].Split('/')[2]);
                        if (annee > 1900) { }
                        else
                        {
                            annee = int.Parse(li[dic["Date"]][6].ToString() + li[dic["Date"]][7].ToString());
                            if (annee > 50) { annee = 1900 + annee; }
                            else { annee = 2000 + annee; }
                        }




                        string date = annee.ToString() + li[dic["Date"]][3] + li[dic["Date"]][4] + li[dic["Date"]][0] + li[dic["Date"]][1];

                        li[dic["Date"]] = date;

                        listWithoutEmpty.Add(li);
                    }
                }

                IOrderedEnumerable<string[]> ordered = listWithoutEmpty.OrderBy(e => int.Parse(e[dic["Date"]])).ThenBy(e => e[dic["HomeTeam"]]);
                int oddsErrorCount = 0;
                float meanOdds = 0;
                float totalOdds = 0;
                int numberOdds = 0;
                foreach (string[] li in ordered)
                {
                    string h = li[dic["BbMxH"]];
                    string d = li[dic["BbMxD"]];
                    string a = li[dic["BbMxA"]];

                    if (!string.IsNullOrEmpty(h) && !string.IsNullOrEmpty(d)
                              && !string.IsNullOrEmpty(a))
                    {
                        float H = float.Parse(h, CultureInfo.InvariantCulture);
                        float D = float.Parse(d, CultureInfo.InvariantCulture);
                        float A = float.Parse(a, CultureInfo.InvariantCulture);
                        float res = 1 / H + 1 / D + 1 / A;
                        numberOdds++;
                        totalOdds += res;
                        if (res < 0.9 && res > 1.1)
                        {
                            oddsErrorCount++;
                        }
                    }
                }
                meanOdds = totalOdds / numberOdds;

                Console.WriteLine("Odds error count : \0");
                Console.WriteLine(oddsErrorCount);

                Console.WriteLine("mean odds : \0");
                Console.WriteLine(meanOdds);

                string[] towrite = new string[list.Count + 1];

                string dicToWrite = "";
                foreach (var i in dic)
                {
                    dicToWrite += i.Key + ",";
                }

                towrite[0] = dicToWrite;

                int cc = 1;
                foreach (string[] li in ordered)
                {
                    string ll = "";
                    foreach (string s in li)
                    {
                        ll += s + ",";
                    }
                    towrite[cc] = ll;
                    cc++;
                }

                File.WriteAllLines(outputPath + country + "TotalNotCrossChecked.txt", towrite);

            }
            Console.WriteLine("finished");
            Console.ReadKey();
        }
    }
}

